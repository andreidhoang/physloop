"""Known answers for actual AlpaSim output extraction (module F2; D-024 completeness gap).

Fixtures: ``tests/fixtures/s2_actual_extract/scenario.json`` (scope public_source_fixture),
hand-written in the pinned AlpaSim e121e591 output shapes; parquet files are written here with
pyarrow in the long layout of ``create_metrics_dataframe`` (data.py:2027-2077; polars writes
large strings). The ledger and unit records are produced with the real producers
(``gate.s2_executor`` events, ``gate.s2_capture.reconcile_unit_attempts``,
``gate.s2_ledger.reconcile_attempts``) exactly as ``scripts/s2_run_units.py`` derives them;
every expected value below is a hand-derived literal. No test calls a process or the network.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA
from gate.s2_actual_extract import assemble_batch, extract_unit
from gate.s2_capture import reconcile_unit_attempts
from gate.s2_diagnostics import describe_validity
from gate.s2_executor import started_event, unit_terminal_events
from gate.s2_extract import PARQUET_COLUMNS
from gate.s2_gate import evaluate_registered
from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_ledger import reconcile_attempts
from scripts import s2_run_units
from scripts.s2_guest_manifest import build_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "s2_actual_extract"
SCENARIO = json.loads((FIXTURES / "scenario.json").read_text())
SCOPE = "public_source_fixture"
UNITS = SCENARIO["units"]
ALL = ("pair_a", "pair_b_late", "invalid_samples", "missing_offroad", "failed_pre_uuid",
       "incomplete_folder", "internal_retry")
U_A = "20000000-0000-4000-8000-0000000000a1"
U_B = "20000000-0000-4000-8000-0000000000b1"
U_RETRY_1 = "20000000-0000-4000-8000-000000000061"
U_RETRY_2 = "20000000-0000-4000-8000-000000000062"
U_INCOMPLETE = "20000000-0000-4000-8000-0000000000f1"
METRIC_NAMES = sorted(SCENARIO["base_streams"])
CONVENTIONS = {
    "schema_version": 1,
    "conventions_id": "fixture-conventions-not-huy",
    "source_ref": "tests/test_s2_actual_extract.py (test fixture, not a registration)",
    "terminal_semantics": "exact_requested_horizon",
    "validity_definition": {"schema_version": 1, "definition_id": "fixture-validity-not-huy",
                            "source_ref": "test fixture",
                            "invalid_when": ["terminal_not_exact_horizon"]},
}
# The wizard writes cfg.eval to <log_dir>/eval-config.yaml (configuration.py:241-249,
# 320-325); the pinned e2e_challenge preset sets fail_on_driver_termination true (dev.yaml:51).
EVAL_CONFIG = ("allow_aggregation_with_failed_rollouts: true\n"
               "fail_on_driver_termination: true\n")
METRIC = {"name": "challenge_scene_score", "kind": "continuous", "higher_is_worse": False,
          "unit": "score_in_0_1"}


# --- fixture producers --------------------------------------------------------------------

def streams_for(spec):
    out = {}
    for name, base in SCENARIO["base_streams"].items():
        if name in spec["drop"]:
            continue
        override = spec["override"].get(name, {})
        values = list(override.get("values", base["values"]))
        valid = list(override.get("valid", [True] * len(values)))
        out[name] = (base["aggregation"], values, valid)
    return out


def parquet_bytes(streams, *, clip, rollout, run_uuid, run_name, values_type=None,
                  timestamps_type=None, run_uuid_column=None):
    """Rows in metric blocks, exactly the pinned writer's columns (data.py:2048-2077)."""
    text = pa.large_string()
    integer_values = values_type is not None and pa.types.is_integer(values_type)
    schema = pa.schema([("name", text), ("timestamps_us", timestamps_type or pa.int64()),
                        ("values", values_type or pa.float64()), ("valid", pa.bool_()),
                        ("time_aggregation", text), ("clipgt_id", text),
                        ("rollout_id", text), ("run_uuid", text), ("run_name", text)])
    assert set(schema.names) == PARQUET_COLUMNS
    rows = []
    for name, (aggregation, values, valid) in streams.items():
        for stamp, value, flag in zip(SCENARIO["timestamps_us"], values, valid, strict=True):
            rows.append({"name": name, "timestamps_us": stamp,
                         "values": int(value) if integer_values else value,
                         "valid": flag, "time_aggregation": aggregation, "clipgt_id": clip,
                         "rollout_id": rollout, "run_uuid": run_uuid_column or run_uuid,
                         "run_name": run_name})
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
    return sink.getvalue().to_pybytes()


def summary_payload(unit, scene):
    """Pinned row shapes: scored rows processing.py:293-337, failure rows failed_rollouts.py."""
    rows = []
    for spec in unit["summary_rows"]:
        ident = {"run_uuid": unit["run_uuid"], "run_name": unit["run_name"],
                 "clipgt_id": scene, "rollout_id": spec["rollout_id"]}
        if spec["kind"] == "runtime_failure":
            ident["run_uuid"] = None
            rows.append({**ident, "status": "fail", "passed": False,
                         "failure_reason": spec["error"], "metrics": {**ident,
                                                                       "error": spec["error"]},
                         "score": 0.0,
                         "score_metrics": {"progress_clipped_rel": None, "progress_rel": None,
                                           "progress_score": 0.0, "collision_at_fault": None,
                                           "offroad": None, "dist_to_gt_trajectory": None,
                                           "lateral_dist_to_gt_trajectory": None,
                                           "left_corridor_laterally": None,
                                           "gt_dist_traveled_m": None}})
            continue
        metrics = {**ident, **SCENARIO["base_summary_metrics"], **spec["metrics_override"]}
        score_metrics = {key: metrics[key] for key in (
            "progress_clipped_rel", "progress_rel", "collision_at_fault", "offroad",
            "dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory",
            "left_corridor_laterally", "gt_dist_traveled_m")}
        score_metrics["progress_score"] = spec["progress_score"]
        rows.append({**ident, "status": spec["status"], "passed": spec["status"] == "pass",
                     "score": spec["score"], "failure_reason": spec["failure_reason"],
                     "metrics": metrics, "score_metrics": score_metrics})
    return {"schema_version": 1, "scene_score_enabled": True, "rollouts": rows,
            "metrics_results": []}


def write_run(root, name, *, scene=None, complete=None, parquet_kwargs=None, streams=None,
              metadata=True, summary=None, eval_config=EVAL_CONFIG):
    unit = UNITS[name]
    scene = scene or unit["job"]["scene_id"]
    run = root / name
    (run / "physloop-capture").mkdir(parents=True)
    (run / "aggregate").mkdir()
    if metadata:
        (run / "run_metadata.yaml").write_text(
            f"run_name: {unit['run_name']}\nrun_uuid: {unit['run_uuid']}\n")
    (run / "physloop-capture" / "compose-up.stdout.txt").write_text("\n".join(unit["log"]) + "\n")
    if eval_config is not None:
        (run / "eval-config.yaml").write_text(eval_config)
    payload = summary_payload(unit, scene) if summary is None else summary
    (run / "aggregate" / "results-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True))
    for uuid, spec in unit["rollouts"].items():
        folder = run / "rollouts" / scene / uuid
        folder.mkdir(parents=True)
        if spec["asl"]:
            folder.joinpath("rollout.asl").write_bytes(b"asl-bytes-not-decoded-here")
        if spec["streams"] is not None:
            folder.joinpath("metrics.parquet").write_bytes(parquet_bytes(
                streams or streams_for(spec["streams"]), clip=scene, rollout=uuid,
                run_uuid=unit["run_uuid"], run_name=unit["run_name"],
                **(parquet_kwargs or {})))
        if (spec["complete"] if complete is None else complete):
            folder.joinpath("_complete").write_bytes(b"")
    # The runner hashes the run dir with symlink_policy="record" (s2_run_units.py:789-802).
    manifest = build_manifest(run, scope=SCOPE, symlink_policy="record")
    assert manifest["status"] == "passed"
    return run, manifest


def runner_report(run, receipts, job, exit_code):
    """The runner's reconcile input (scripts/s2_run_units.py _Run._record_attempts)."""
    by_path = {item["path"]: item for item in receipts}
    logs = []
    path = "physloop-capture/compose-up.stdout.txt"
    if path in by_path:
        logs.append({"source": "compose_up_stdout",
                     "text": (run / path).read_bytes().decode("utf-8", "replace")})
    metadata = (run / "run_metadata.yaml").read_text() if "run_metadata.yaml" in by_path \
        else None
    summary = json.loads((run / "aggregate/results-summary.json").read_text())
    evidence = {"scope": SCOPE, "scene_id": job["scene_id"],
                "requested_steps": job["requested_steps"], "logs": logs,
                "rollout_files": sorted(by_path), "run_metadata": metadata, "summary": summary,
                "compose_exit_code": exit_code}
    return reconcile_unit_attempts(evidence), len(logs)


def unit_record(index, job, report, ref, terminal, manifest, exit_code, runner):
    """A unit record in the exact ``_Run._envelope`` shape (scripts/s2_run_units.py:614-620).

    ``runner`` overrides the recorder state: ``outcome``/``stop_run`` and whether the
    ``compose_up`` step was recorded (absent when the Compose call itself raised) and
    ``timed_out`` (s2_run_units.py:741-755).
    """
    steps = []
    if runner.get("compose_up", True):
        steps.append({"id": "compose_up", "status": "unverified", "outcome": "recorded",
                      "returncode": exit_code, "timed_out": runner.get("timed_out", False),
                      "exit_code_is_success_evidence": False})
    steps += [{"id": "run_dir_hash", "status": manifest["status"], "files": manifest["totals"],
               "files_sha256": manifest["files_sha256"], "issues": []},
              {"id": "reconcile", "status": "passed", "outcome": "recorded",
               "unit_outcome": report["unit_outcome"], "evidence_ref": ref},
              {"id": "ledger_terminal", "status": "passed",
               "attempts": [{"attempt_id": e["attempt_id"], "type": e["type"],
                             "reason": e["reason"]}
                            for e in terminal if e["type"] != "started"]}]
    return {"artifact_kind": "s2_unit_record", "schema_version": 1, "scope": SCOPE,
            "producer": {"module": "scripts.s2_run_units", "function": "_Run.unit",
                         "source_sha256": None},
            "input_hashes": {}, "units": {"timeout_s": "seconds",
                                          "timestamp_us": "monotonic microseconds"},
            "status": "passed", "status_reasons": [],
            "index": index, "job_id": job["job_id"], "scene_id": job["scene_id"],
            "steps": steps, "outcome": runner.get("outcome", "executed"),
            "stop_run": runner.get("stop_run"), "compose_exit_code": exit_code,
            "unit_outcome": report["unit_outcome"], "ledger_events": 1 + len(terminal),
            "issues": []}


def build(tmp_path, names, *, scene_override=None, runner=None, **write_kwargs):
    """Run dirs, receipts, unit records and one ledger over ``names`` (runner semantics).

    ``runner[name]`` may set ``interruption`` (the recorder's ``state["interruption"]``,
    s2_run_units.py:537/750/960), ``compose_exit_code`` and the ``unit_record`` overrides.
    """
    schedule, events, units = [], [], {}
    clock = 1000
    for name in names:
        unit = UNITS[name]
        job = {**unit["job"], "phase": "harmonizer_pilot",
               "protocol_hash": SCENARIO["protocol_hash"],
               "requested_steps": SCENARIO["requested_steps"]}
        if scene_override and name in scene_override:
            job["scene_id"] = scene_override[name]
        schedule.append(job)
        run, manifest = write_run(tmp_path, name, scene=unit["job"]["scene_id"],
                                  **write_kwargs.get(name, {}))
        receipts = manifest["files"]
        state = (runner or {}).get(name, {})
        exit_code = state.get("compose_exit_code", unit["compose_exit_code"])
        report, log_count = runner_report(run, receipts, job, exit_code)
        ref = f"unit_attempt_evidence:{identity_hash(report)}"
        events.append(started_event(job["job_id"], 1, clock, [f"packet_unit:{name}"]))
        terminal = unit_terminal_events(job["job_id"], report, clock + 1, ref,
                                        log_count=log_count,
                                        interruption=state.get("interruption"))
        events.extend(terminal)
        clock += 10
        record = unit_record(len(units), job, report, ref, terminal, manifest, exit_code,
                             state)
        units[name] = (run, receipts, record)
    return units, reconcile_attempts(schedule, events)


def extract(units, ledger, name, **kwargs):
    run, receipts, record = units[name]
    return extract_unit(run, receipts, record, ledger, scope=SCOPE, **kwargs)


def codes(artifact):
    return {item["code"]: item["severity"] for item in artifact["issues"]}


# --- fixture integrity --------------------------------------------------------------------

def test_fixture_files_match_their_provenance():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert provenance["scope"] == SCOPE == SCENARIO["scope"]
    assert set(provenance["files"]) == {"README.md", "scenario.json"}
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_module_makes_no_process_or_network_import():
    source = (Path(__file__).parents[1] / "gate" / "s2_actual_extract.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"hashlib", "math", "pathlib", "pyarrow", "yaml", "gate"}


# --- scope refusals -------------------------------------------------------------------------

def test_synthetic_and_unknown_scopes_are_refused(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    for scope in ("synthetic", None, "fixture"):
        with pytest.raises(ValueError):
            extract_unit(run, receipts, record, ledger, scope=scope)
        with pytest.raises(ValueError):
            assemble_batch([], ledger, scope=scope, observed_evaluator_sha="x",
                           conventions=None)
    synthetic = copy.deepcopy(ledger)
    synthetic["jobs"][0]["phase"] = "synthetic"
    with pytest.raises(ValueError, match="synthetic inputs are refused"):
        extract_unit(run, receipts, record, synthetic, scope=SCOPE)
    with pytest.raises(ValueError, match="scope"):
        extract_unit(run, receipts, {**record, "scope": "actual"}, ledger, scope=SCOPE)


@pytest.mark.parametrize("change", [
    {"artifact_kind": "s2_diagnostic_unit_record"},
    {"role": "engineering_diagnostic"},
    {"inference_input": False},
])
def test_diagnostic_unit_records_are_refused_by_name(tmp_path, change):
    """G2-04: a diagnostic unit never becomes an inference observation; the refusal names it."""
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    with pytest.raises(ValueError, match="diagnostic"):
        extract_unit(run, receipts, {**record, **change}, ledger, scope=SCOPE)


def test_diagnostic_qualification_ledger_jobs_are_refused_by_name(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    diagnostic = copy.deepcopy(ledger)
    diagnostic["jobs"][0]["phase"] = "diagnostic_qualification"
    with pytest.raises(ValueError, match="diagnostic"):
        extract_unit(run, receipts, record, diagnostic, scope=SCOPE)
    with pytest.raises(ValueError, match="diagnostic"):
        assemble_batch([], diagnostic, scope=SCOPE, observed_evaluator_sha="x",
                       conventions=None)


# --- per-unit known answers -----------------------------------------------------------------

def test_clean_unit_observation_known_answer(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "passed" and codes(unit) == {}
    assert unit["attempt_evidence"]["bound_to_unit_record"] is True
    assert unit["attempt_evidence"]["bound_to_ledger"] is True
    assert unit["attempt_evidence"]["unit_outcome"] == "complete"
    [row] = unit["observations"]
    provenance = row.pop("provenance_ref")
    assert provenance.startswith("s2_actual_extract:") and len(provenance) == 18 + 64
    assert row == {
        "attempt_id": "job-pair-a/attempt-1", "job_id": "job-pair-a",
        "run_uuid": "11111111-1111-4111-8111-1111111111a1", "clipgt_id": "clip-s1",
        "rollout_id": U_A, "completion_validated": True, "required_metrics_verified": True,
        "full_max_deviation_m": 0.8, "requested_steps": 3, "actual_steps": 3,
        "metric_names": METRIC_NAMES, "raw_metric_rows": 44,
        "run_name": "e2e_challenge_dev", "corridor_departure": False,
        "black_frame_count": 0}
    assert unit["counts"] == {"ledger_attempts": 1, "observations": 1,
                              "attempts_without_upstream_identity": 0, "invalid_samples": 0,
                              "summary_rows": 1, "summary_rows_null_run_uuid": 0,
                              "folders_without_complete": 0}
    assert unit["normalize_inputs"]["identity_map"] == [{
        "run_uuid": "11111111-1111-4111-8111-1111111111a1", "clipgt_id": "clip-s1",
        "rollout_id": U_A, "scene_id": "clip-s1", "road_id": "road-1", "stratum": "urban",
        "replicate_id": 0, "policy_id": "fixture-policy-a"}]
    assert unit["score_admitted"] is False and unit["runtime_qualified"] is False


def test_late_deviation_uses_the_full_stream_not_the_truncated_summary(tmp_path):
    units, ledger = build(tmp_path, ["pair_b_late"])
    unit = extract(units, ledger, "pair_b_late")
    assert unit["status"] == "passed"
    [row] = unit["observations"]
    # Summary dist_to_gt_trajectory is 1.0 (rows kept up to the collision at 200000 us,
    # modifiers.py:99-101); the full stream [0.5, 1.0, 2.0, 4.2] has maximum 4.2.
    assert unit["normalize_inputs"]["summary"]["rollouts"][0]["metrics"][
        "dist_to_gt_trajectory"] == 1.0
    assert row["full_max_deviation_m"] == 4.2
    assert unit["out_of_envelope"] == {"threshold_m": 4.0,
                                       "attempt_ids": ["job-pair-b/attempt-1"],
                                       "comparison": "full_max_deviation_m >= threshold_m"}
    assert unit["attempts"][0]["out_of_envelope"] is True
    assert codes(unit) == {"summary_deviation_below_full_stream": "info",
                           "out_of_envelope_attempts": "info"}


def test_invalid_samples_are_preserved_counted_and_block_metric_verification(tmp_path):
    units, ledger = build(tmp_path, ["invalid_samples"])
    unit = extract(units, ledger, "invalid_samples")
    assert unit["status"] == "passed"
    metrics = unit["attempts"][0]["metrics"]
    assert metrics["invalid_samples"] == {
        "total": 2, "by_stream": {"collision_front": 1, "img_is_black": 1},
        "required": {"collision_front": 1}, "valid_null_by_stream": {}}
    assert metrics["streams"]["collision_front"]["rows"] == 4
    assert metrics["streams"]["collision_front"]["valid_false"] == 1
    assert metrics["required_defects"] == ["valid_false:collision_front"]
    [row] = unit["observations"]
    assert row["required_metrics_verified"] is False
    assert row["completion_validated"] is True and row["full_max_deviation_m"] == 0.8
    assert row["raw_metric_rows"] == 44
    assert unit["counts"]["invalid_samples"] == 2
    assert codes(unit) == {"invalid_samples_preserved": "info"}


def test_missing_offroad_is_never_zero_filled(tmp_path):
    units, ledger = build(tmp_path, ["missing_offroad"])
    unit = extract(units, ledger, "missing_offroad")
    metrics = unit["attempts"][0]["metrics"]
    assert metrics["required_defects"] == ["missing:offroad"]
    assert "offroad" not in metrics["streams"]
    assert metrics["required_streams"]["missing"] == ["offroad"]
    [row] = unit["observations"]
    assert row["required_metrics_verified"] is False
    assert "offroad" not in row["metric_names"] and row["raw_metric_rows"] == 40
    assert codes(unit) == {"summary_metric_without_raw_stream": "warning"}
    [warning] = unit["issues"]
    assert warning["path"] == "attempt:job-no-offroad/attempt-1.offroad"


def test_pre_uuid_failure_row_is_counted_without_an_invented_identity(tmp_path):
    units, ledger = build(tmp_path, ["failed_pre_uuid"])
    unit = extract(units, ledger, "failed_pre_uuid")
    assert unit["status"] == "passed" and unit["observations"] == []
    assert [(a["attempt_id"], a["ledger_state"], a["terminal_reason"], a["upstream_identity"])
            for a in unit["attempts"]] == [
        ("job-pre-uuid/attempt-1", "failed",
         "upstream_failure_before_rollout_uuid:retryable=False", None)]
    assert unit["summary_rows_null_run_uuid"] == ["failed-0"]
    assert unit["counts"]["attempts_without_upstream_identity"] == 1
    assert unit["normalize_inputs"]["identity_map"] == []
    assert unit["normalize_inputs"]["summary"]["rollouts"][0]["run_uuid"] is None
    assert codes(unit) == {"attempts_without_upstream_identity": "info"}


def test_folder_without_complete_and_exit_zero_is_not_completion(tmp_path):
    units, ledger = build(tmp_path, ["incomplete_folder"])
    unit = extract(units, ledger, "incomplete_folder")
    assert units["incomplete_folder"][2]["compose_exit_code"] == 0
    assert unit["attempt_evidence"]["compose_exit_code_used_for_success"] is False
    [row] = unit["observations"]
    assert row == {**{k: row[k] for k in ("provenance_ref",)},
                   "attempt_id": "job-incomplete/attempt-1", "job_id": "job-incomplete",
                   "run_uuid": "11111111-1111-4111-8111-1111111111f1",
                   "clipgt_id": "clip-s5", "rollout_id": U_INCOMPLETE,
                   "completion_validated": False, "required_metrics_verified": False,
                   "full_max_deviation_m": None, "requested_steps": 3, "actual_steps": 3}
    assert unit["attempts"][0]["deviation_unknown_reason"] == "no_metrics_file"
    assert unit["attempts"][0]["terminal_reason"] == "upstream_rollout_failed:retryable=False"
    assert unit["counts"]["folders_without_complete"] == 1
    assert unit["summary_rows_null_run_uuid"] == [U_INCOMPLETE]


def test_internal_retry_keeps_both_rollout_uuids(tmp_path):
    units, ledger = build(tmp_path, ["internal_retry"])
    unit = extract(units, ledger, "internal_retry")
    assert unit["status"] == "passed"
    assert unit["attempt_evidence"]["internal_retries"] == 1
    assert [(r["attempt_id"], r["rollout_id"], r["completion_validated"],
             r["required_metrics_verified"], r["full_max_deviation_m"])
            for r in unit["observations"]] == [
        ("job-retry/attempt-1", U_RETRY_1, False, False, None),
        ("job-retry/attempt-2", U_RETRY_2, True, True, 0.8)]
    assert [a["ledger_state"] for a in unit["attempts"]] == ["failed", "completed"]
    assert unit["counts"]["folders_without_complete"] == 1
    assert ledger["admitted"] is False
    assert "retry_selection_unregistered" in {item["code"] for item in ledger["issues"]}


def test_extraction_is_deterministic_and_receipt_order_invariant(tmp_path):
    units, ledger = build(tmp_path, ["pair_b_late"])
    run, receipts, record = units["pair_b_late"]
    first = extract_unit(run, receipts, record, ledger, scope=SCOPE)
    second = extract_unit(run, list(reversed(receipts)), record, ledger, scope=SCOPE)
    assert identity_hash(first) == identity_hash(second)


# --- per-unit negative controls -------------------------------------------------------------

def test_tampered_metrics_bytes_block(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    run = units["pair_a"][0]
    (run / "rollouts" / "clip-s1" / U_A / "metrics.parquet").write_bytes(b"tampered")
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked" and codes(unit)["receipt_mismatch"] == "blocking"
    assert unit["observations"][0]["required_metrics_verified"] is False
    assert unit["observations"][0]["full_max_deviation_m"] is None


def test_unit_record_evidence_reference_must_match_receipted_bytes(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    record = copy.deepcopy(record)
    [step] = [item for item in record["steps"] if item["id"] == "reconcile"]
    step["evidence_ref"] = "unit_attempt_evidence:" + "0" * 64
    unit = extract_unit(run, receipts, record, ledger, scope=SCOPE)
    assert unit["status"] == "blocked"
    assert codes(unit)["attempt_evidence_unbound"] == "blocking"


def test_ledger_identity_must_equal_the_evidence(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    ledger = copy.deepcopy(ledger)
    ledger["attempts"][0]["upstream_identity"]["run_uuid"] = (
        "99999999-9999-4999-8999-999999999999")
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert {"ledger_attempts_differ_from_evidence", "identity_run_uuid_mismatch"} <= set(
        codes(unit))


def test_scene_must_equal_clipgt_id(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"], scene_override={"pair_a": "clip-other"})
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert codes(unit)["scene_clip_mismatch"] == "blocking"
    assert unit["attempts"][0]["ledger_state"] == "failed"
    assert unit["observations"][0]["completion_validated"] is False


def test_metrics_identity_and_schema_mismatches_block(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"parquet_kwargs": {
        "run_uuid_column": "99999999-9999-4999-8999-999999999999"}})
    unit = extract(units, ledger, "pair_a")
    assert codes(unit)["metrics_identity_mismatch"] == "blocking"
    assert unit["observations"][0]["full_max_deviation_m"] is None
    units, ledger = build(tmp_path / "ints", ["pair_a"],
                          pair_a={"parquet_kwargs": {"values_type": pa.int64()}})
    unit = extract(units, ledger, "pair_a")
    assert codes(unit)["metrics_schema_differs"] == "blocking"
    assert unit["observations"][0]["required_metrics_verified"] is False


def test_missing_complete_marker_is_not_completion_despite_log_and_summary(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"complete": False})
    unit = extract(units, ledger, "pair_a")
    [row] = unit["observations"]
    assert row["completion_validated"] is False
    assert unit["attempts"][0]["terminal_reason"] == "upstream_attempt_incomplete"


def test_nonfinite_deviation_is_unknown_never_zero(tmp_path):
    streams = streams_for({"drop": [], "override": {
        "dist_to_gt_trajectory": {"values": [0.1, float("nan"), 0.2, 0.3]}}})
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"streams": streams})
    unit = extract(units, ledger, "pair_a")
    [row] = unit["observations"]
    assert row["full_max_deviation_m"] is None and row["required_metrics_verified"] is False
    assert unit["attempts"][0]["deviation_unknown_reason"] == (
        "null_nonfinite_or_negative_sample")
    assert "values_nonfinite:dist_to_gt_trajectory" in unit["attempts"][0]["metrics"][
        "required_defects"]


def test_receipts_must_be_the_pre_teardown_manifest(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    run, _receipts, record = units["pair_a"]
    (run / "physloop-capture" / "unit-attempt-evidence.json").write_text("{}")
    rewalked = build_manifest(run, scope=SCOPE, symlink_policy="record")["files"]
    unit = extract_unit(run, rewalked, record, ledger, scope=SCOPE)
    assert unit["status"] == "blocked"
    assert codes(unit)["attempt_evidence_unbound"] == "blocking"


def test_malformed_ledger_attempt_is_refused(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    broken = copy.deepcopy(ledger)
    del broken["attempts"][0]["events"]
    with pytest.raises(ValueError, match="ledger attempts"):
        extract(units, broken, "pair_a")


def test_missing_run_metadata_blocks(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"metadata": False})
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert codes(unit)["run_metadata_absent"] == "blocking"


# --- batch: consumers ---------------------------------------------------------------------

def batch(tmp_path, names, conventions, **kwargs):
    units, ledger = build(tmp_path, names)
    extractions = [extract(units, ledger, name) for name in names]
    return assemble_batch(extractions, ledger, scope=SCOPE,
                          observed_evaluator_sha=kwargs.get("sha", SUPPORTED_EVALUATOR_SHA),
                          conventions=conventions), extractions, ledger


def test_full_batch_is_blocked_but_every_fact_is_kept(tmp_path):
    result, _, _ = batch(tmp_path, ALL, None)
    assert result["admission"]["status"] == "blocked"
    assert result["admission"]["code"] == "owner_conventions_unregistered"
    assert result["admission"]["reasons"] == [
        "owner_conventions_unregistered", "ledger_not_admitted",
        "normalization_not_admitted:job-incomplete",
        "normalization_not_admitted:job-invalid",
        "normalization_not_admitted:job-no-offroad",
        "normalization_not_admitted:job-pre-uuid",
        "normalization_not_admitted:job-retry",
        "validity_report_blocked"]
    assert result["gate_inputs"] == {"records": [], "adapter_metric": None}
    assert [row["attempt_id"] for row in result["observations"]] == [
        "job-pair-a/attempt-1", "job-pair-b/attempt-1", "job-invalid/attempt-1",
        "job-no-offroad/attempt-1", "job-incomplete/attempt-1", "job-retry/attempt-1",
        "job-retry/attempt-2"]
    assert result["out_of_envelope"]["attempt_ids"] == ["job-pair-b/attempt-1"]
    validity = result["validity_report"]
    assert validity["deviation_bins"]["counts"] == {
        "[0,1)": 3, "[1,2)": 0, "[2,4)": 0, "[4,inf)": 1, "unknown": 3}
    assert validity["out_of_envelope"]["job_ids"] == ["job-pair-b"]
    assert validity["denominators"]["bound_observations"] == 7
    assert validity["denominators"]["unbound_observations"] == 0
    observation_codes = {item["code"] for item in validity["issues"]}
    assert not observation_codes & {"unknown_observation_field", "invalid_observation",
                                    "invalid_observation_value",
                                    "observation_identity_mismatch",
                                    "observation_requested_steps_mismatch"}
    adapter_codes = {
        item["job_id"]: sorted({i["code"] for i in item["normalization"]["issues"]})
        for item in result["normalizations"]}
    assert adapter_codes == {
        "job-pair-a": [], "job-pair-b": [],
        "job-invalid": ["metrics_unverified"],
        "job-no-offroad": ["metrics_unverified"],
        "job-pre-uuid": ["invalid_identity"],
        # The failure row's null run_uuid cannot key the bound identity, so the bound
        # identity is missing from the summary and its observation is unmatched.
        "job-incomplete": ["invalid_identity", "missing_expected_rollout",
                           "unmatched_observation"],
        # The failed first rollout has no summary row (never in summary, lane R2 C2).
        "job-retry": ["missing_expected_rollout", "unmatched_observation"]}


def test_admitted_records_feed_the_registered_gate(tmp_path):
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], CONVENTIONS)
    assert result["admission"] == {"status": "admitted", "code": None, "reasons": []}
    assert result["status"] == "passed" and result["score_admitted"] is True
    assert result["conventions"]["status"] == "supplied_not_authenticated"
    records = result["gate_inputs"]["records"]
    assert records == [
        {"arm": "A", "scene_id": "clip-s1", "road_id": "road-1", "stratum": "urban",
         "replicate_id": 0, "value": 0.8, "completed": True, "deviation_m": 0.8,
         "seed": None},
        {"arm": "B", "scene_id": "clip-s1", "road_id": "road-1", "stratum": "urban",
         "replicate_id": 0, "value": 0.0, "completed": True, "deviation_m": 4.2,
         "seed": None}]
    assert result["gate_inputs"]["adapter_metric"] == METRIC
    assert result["out_of_envelope"]["records"] == 1
    validity = result["validity_report"]
    assert validity["status"] == "passed"
    assert validity["validity_table"]["table"] == {
        "A_valid_B_valid": 1, "A_valid_B_invalid": 0, "A_invalid_B_valid": 0,
        "A_invalid_B_invalid": 0}
    registration = {
        "schema_version": 1, "registration_id": "fixture-registration-not-huy",
        "reference": "fixture-policy-a", "candidate": "fixture-policy-b",
        "protocol": {"identity": "fixture-protocol", "config_hash": "0" * 64,
                     "evaluator": SUPPORTED_EVALUATOR_SHA},
        "seed_mode": "unseeded", "score_definition": "fixture score definition",
        "deviation_convention": "fixture deviation convention",
        "estimand": "equal road / equal scene within road / equal replicate within scene",
        "required_strata": ["urban"], "margin": 0.10, "alpha": 0.05, "metric": METRIC,
        "analysis_mode": "fixed_sample"}
    gate = evaluate_registered(records, registration, scope=SCOPE,
                               adapter_metric=result["gate_inputs"]["adapter_metric"])
    assert gate["status"] == "passed"
    assert gate["verdict"] == "INSUFFICIENT"
    assert gate["out_of_envelope"]["count"] == 1
    assert gate["out_of_envelope"]["rollouts"] == [
        {"arm": "B", "scene_id": "clip-s1", "road_id": "road-1", "stratum": "urban",
         "replicate_id": 0, "deviation_m": 4.2}]


def test_without_conventions_the_same_batch_stays_blocked(tmp_path):
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], None)
    assert result["admission"] == {"status": "blocked",
                                   "code": "owner_conventions_unregistered",
                                   "reasons": ["owner_conventions_unregistered"]}
    assert result["gate_inputs"]["records"] == []
    assert all(item["normalization"]["admitted"] for item in result["normalizations"])
    assert result["validity_report"]["validity_table"]["code"] == (
        "validity_definition_unregistered")
    assert result["conventions"] is None


@pytest.mark.parametrize(("change", "code"), [
    ({"terminal_semantics": "accept_clipped_horizon"}, "terminal_semantics_unsupported"),
    ({"terminal_semantics": "anything"}, "invalid_owner_conventions"),
    ({"schema_version": 2}, "invalid_owner_conventions"),
    ({"extra": True}, "invalid_owner_conventions"),
    ({"validity_definition": {"schema_version": 1, "definition_id": "x", "source_ref": "y",
                              "invalid_when": ["not_an_indicator"]}},
     "invalid_owner_conventions"),
])
def test_unusable_conventions_block(tmp_path, change, code):
    conventions = {**CONVENTIONS, **change}
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], conventions)
    assert result["admission"]["status"] == "blocked"
    assert result["admission"]["code"] == code
    assert result["gate_inputs"]["records"] == []


@pytest.mark.parametrize(("invalid_when", "reasons"), [
    (["out_of_envelope"], ["unit_validity_invalid:job-pair-b"]),
    (["render_incomplete"], ["unit_validity_unknown:job-pair-a",
                             "unit_validity_unknown:job-pair-b"]),
])
def test_units_not_valid_under_the_definition_block_without_exclusion(
        tmp_path, invalid_when, reasons):
    conventions = copy.deepcopy(CONVENTIONS)
    conventions["validity_definition"]["invalid_when"] = invalid_when
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], conventions)
    assert result["admission"]["reasons"] == reasons
    assert result["gate_inputs"]["records"] == []
    assert len(result["observations"]) == 2


def test_unsupported_evaluator_blocks(tmp_path):
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], CONVENTIONS, sha="f" * 40)
    # Rejected normalizations are also records_on_rejected_batch for describe_validity.
    assert result["admission"]["reasons"] == ["normalization_not_admitted:job-pair-a",
                                              "normalization_not_admitted:job-pair-b",
                                              "validity_report_blocked"]
    assert {i["code"] for item in result["normalizations"]
            for i in item["normalization"]["issues"]} == {"unsupported_evaluator"}


def test_batch_needs_exactly_one_extraction_per_scheduled_job(tmp_path):
    units, ledger = build(tmp_path, ["pair_a", "pair_b_late"])
    first = extract(units, ledger, "pair_a")
    result = assemble_batch([first, first], ledger, scope=SCOPE,
                            observed_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
                            conventions=CONVENTIONS)
    assert {item["code"] for item in result["issues"]} == {"duplicate_unit_extraction",
                                                           "unit_extraction_missing"}
    assert "batch_inputs_inconsistent" in result["admission"]["reasons"]
    assert result["gate_inputs"]["records"] == []
    with pytest.raises(ValueError):
        assemble_batch([{"artifact_kind": "other"}], ledger, scope=SCOPE,
                       observed_evaluator_sha=SUPPORTED_EVALUATOR_SHA, conventions=None)
    with pytest.raises(ValueError):
        assemble_batch([first], ledger, scope=SCOPE, observed_evaluator_sha="",
                       conventions=None)


def test_observations_are_exactly_the_validity_consumer_shape(tmp_path):
    units, ledger = build(tmp_path, ALL)
    rows = [row for name in ALL for row in extract(units, ledger, name)["observations"]]
    report = describe_validity(ledger, rows, scope=SCOPE)
    assert report["denominators"]["bound_observations"] == len(rows) == 7
    assert report["status"] == "passed"


# --- review regressions (F2-1 .. F2-10) ---------------------------------------------------

def rewrite_pair_b_metrics(run, maximum):
    streams = streams_for({"drop": [], "override": {
        "dist_to_gt_trajectory": {"values": [0.5, 1.0, 2.0, maximum]}}})
    unit = UNITS["pair_b_late"]
    (run / "rollouts" / "clip-s1" / U_B / "metrics.parquet").write_bytes(parquet_bytes(
        streams, clip="clip-s1", rollout=U_B, run_uuid=unit["run_uuid"],
        run_name=unit["run_name"]))


def test_metrics_bytes_swapped_at_an_existing_path_block(tmp_path):
    # F2-1: same path, new bytes, receipts re-walked: the 4.2 m excursion must not vanish.
    units, ledger = build(tmp_path, ["pair_b_late"])
    run, receipts, record = units["pair_b_late"]
    rewrite_pair_b_metrics(run, 3.9)
    rewalked = build_manifest(run, scope=SCOPE, symlink_policy="record")["files"]
    assert [item["path"] for item in rewalked] == [item["path"] for item in receipts]
    unit = extract_unit(run, rewalked, record, ledger, scope=SCOPE)
    assert unit["status"] == "blocked"
    assert codes(unit)["run_dir_manifest_unbound"] == "blocking"
    assert unit["run_dir_manifest"] == {
        "receipts_files_sha256": identity_hash(rewalked),
        "unit_record_files_sha256": record["steps"][1]["files_sha256"],
        "unit_record_status": "passed", "bound": False}
    original = extract_unit(run, receipts, record, ledger, scope=SCOPE)
    assert original["run_dir_manifest"]["bound"] is True
    assert codes(original)["receipt_mismatch"] == "blocking"


@pytest.mark.parametrize("change", ["drop", "status", "duplicate"])
def test_run_dir_hash_step_must_bind_the_receipts(tmp_path, change):
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    record = copy.deepcopy(record)
    [step] = [item for item in record["steps"] if item["id"] == "run_dir_hash"]
    if change == "drop":
        record["steps"].remove(step)
    elif change == "status":
        step["status"] = "blocked"
    else:
        record["steps"].append(dict(step))
    unit = extract_unit(run, receipts, record, ledger, scope=SCOPE)
    assert unit["status"] == "blocked"
    assert codes(unit)["run_dir_manifest_unbound"] == "blocking"
    assert unit["run_dir_manifest"]["bound"] is False


def test_batch_artifacts_are_canonical_json(tmp_path):
    # F2-2: a blocked batch (tuple identities in adapter issues) must still hash.
    blocked, extractions, _ = batch(tmp_path / "all", ALL, None)
    assert blocked["admission"]["status"] == "blocked"
    canonical_json_bytes(blocked)
    assert len(identity_hash(blocked)) == 64
    identities = [issue["identity"] for item in blocked["normalizations"]
                  for issue in item["normalization"]["issues"] if "identity" in issue]
    assert identities and all(type(value) in (list, type(None)) for value in identities)
    for item in extractions:
        canonical_json_bytes(item)
    admitted, _, _ = batch(tmp_path / "pair", ["pair_a", "pair_b_late"], CONVENTIONS)
    assert admitted["admission"]["status"] == "admitted"
    assert len(identity_hash(admitted)) == 64


def test_summary_deviation_above_the_full_stream_blocks(tmp_path, monkeypatch):
    # F2-3: upstream aggregates a subset of the same samples, so a larger summary value
    # means parquet and summary disagree.
    unit = copy.deepcopy(UNITS["pair_a"])
    unit["summary_rows"][0]["metrics_override"]["dist_to_gt_trajectory"] = 5.0
    monkeypatch.setitem(UNITS, "pair_a", unit)
    units, ledger = build(tmp_path, ["pair_a"])
    extraction = extract(units, ledger, "pair_a")
    assert extraction["status"] == "blocked"
    assert codes(extraction)["summary_deviation_above_full_stream"] == "blocking"
    assert extraction["observations"][0]["full_max_deviation_m"] == 0.8


@pytest.mark.parametrize("kwargs", [{"values_type": pa.float32()},
                                    {"timestamps_type": pa.int32()}])
def test_arrow_types_must_be_the_pinned_writer_types(tmp_path, kwargs):
    # F2-4: data.py:2056 writes float64 values; timestamps int64; valid bool.
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"parquet_kwargs": kwargs})
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert codes(unit)["metrics_schema_differs"] == "blocking"
    assert unit["observations"][0]["full_max_deviation_m"] is None
    assert unit["observations"][0]["required_metrics_verified"] is False


@pytest.mark.parametrize(("state", "reason"), [
    ({"interruption": "compose_up_timeout", "timed_out": True}, "compose_up_timeout"),
    ({"interruption": "internal_error:RuntimeError", "compose_up": False,
      "compose_exit_code": None, "outcome": "internal_error",
      "stop_run": "internal_error:RuntimeError"}, "internal_error:RuntimeError"),
    ({"interruption": "interrupted:KeyboardInterrupt", "compose_up": False,
      "compose_exit_code": None, "outcome": "interrupted",
      "stop_run": "interrupted:KeyboardInterrupt"}, "interrupted:KeyboardInterrupt"),
])
def test_runner_cancellation_is_rederived_from_the_unit_record(tmp_path, state, reason):
    # F2-5: the runner passes state["interruption"] (s2_run_units.py:959-960).
    units, ledger = build(tmp_path, ["pair_a"], runner={"pair_a": state},
                          pair_a={"complete": False})
    assert [(a["state"], a["terminal_reason"]) for a in ledger["attempts"]] == [
        ("cancelled", reason)]
    unit = extract(units, ledger, "pair_a")
    assert "ledger_attempts_differ_from_evidence" not in codes(unit)
    assert unit["attempt_evidence"]["bound_to_ledger"] is True
    assert unit["attempt_evidence"]["runner_interruption"] == reason
    assert unit["observations"][0]["completion_validated"] is False


def test_interrupt_after_compose_returned_is_not_a_cancellation(tmp_path):
    # stopped_compose=False (captures/teardown): no interruption passed (s2_run_units.py:474-486).
    units, ledger = build(tmp_path, ["pair_a"], runner={"pair_a": {
        "outcome": "interrupted", "stop_run": "interrupted:KeyboardInterrupt"}})
    unit = extract(units, ledger, "pair_a")
    assert unit["attempt_evidence"]["runner_interruption"] is None
    assert unit["attempt_evidence"]["bound_to_ledger"] is True
    assert codes(unit) == {}


def test_summary_without_a_rollouts_list_blocks(tmp_path):
    # F2-6: valid JSON that is not schema_version 1 with a rollouts list.
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"summary": []})
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert codes(unit)["summary_invalid"] == "blocking"
    assert "invalid_summary" in unit["attempt_evidence"]["codes"]


def test_read_bounds_equal_the_runner_bounds(tmp_path):
    # F2-7: same bound per file class as scripts/s2_run_units.py:159-160.
    import gate.s2_actual_extract as module
    assert module.MAX_LOG_BYTES == s2_run_units.MAX_LOG_BYTES == 512 * 1024 * 1024
    assert module.MAX_CONFIG_BYTES == s2_run_units.MAX_CONFIG_BYTES == 16 * 1024 * 1024
    units, ledger = build(tmp_path, ["pair_a"])
    unit = extract(units, ledger, "pair_a")
    assert unit["bounds"] == {"max_parquet_bytes": 256 * 1024 * 1024,
                              "max_log_bytes": 512 * 1024 * 1024,
                              "max_config_bytes": 16 * 1024 * 1024}
    # run_metadata.yaml is read under the config bound, logs under the log bound.
    small = extract(units, ledger, "pair_a", max_config_bytes=10)
    assert {item["path"] for item in small["issues"]
            if item["code"] == "file_exceeds_bound"} == {"eval-config.yaml",
                                                         "run_metadata.yaml"}


@pytest.mark.parametrize("change", ["no_compose_exit_code", "extra_key", "bool_exit_code"])
def test_malformed_unit_record_raises_value_error(tmp_path, change):
    # F2-8: structural misuse is ValueError, never KeyError.
    units, ledger = build(tmp_path, ["pair_a"])
    run, receipts, record = units["pair_a"]
    record = copy.deepcopy(record)
    if change == "no_compose_exit_code":
        del record["compose_exit_code"]
    elif change == "extra_key":
        record["extra"] = 1
    else:
        record["compose_exit_code"] = True
    with pytest.raises(ValueError, match="unit_record"):
        extract_unit(run, receipts, record, ledger, scope=SCOPE)


@pytest.mark.parametrize("key", ["scene_id", "requested_steps", "selected_attempt_id"])
def test_malformed_ledger_job_raises_value_error(tmp_path, key):
    units, ledger = build(tmp_path, ["pair_a"])
    broken = copy.deepcopy(ledger)
    del broken["jobs"][0][key]
    with pytest.raises(ValueError, match="ledger jobs"):
        extract(units, broken, "pair_a")


@pytest.mark.parametrize(("eval_config", "code", "flag"), [
    (None, "eval_config_absent", None),
    ("fail_on_driver_termination: false\n", "driver_termination_not_fatal", False),
    ("allow_aggregation_with_failed_rollouts: true\n", "driver_termination_not_fatal", None),
])
def test_horizon_is_withheld_unless_driver_termination_fails_the_rollout(
        tmp_path, eval_config, code, flag):
    # F2-9: with fail_on_driver_termination false (EvalConfig default) a driver-ended
    # session still writes _complete and COMPLETED, so steps= is only the planned horizon.
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"eval_config": eval_config})
    unit = extract(units, ledger, "pair_a")
    assert codes(unit) == {code: "warning"}
    [row] = unit["observations"]
    assert "requested_steps" not in row and "actual_steps" not in row
    assert unit["horizon"]["fail_on_driver_termination"] is flag
    assert unit["horizon"]["steps_reported"] is False


@pytest.mark.parametrize(("eval_config", "error"), [
    ("a: [unclosed\n", "ParserError"),
    # Parses, but the constructor raises: month 13 (ValueError from datetime.date).
    ("fail_on_driver_termination: true\nwhen: 2026-13-45\n", "ValueError"),
    # Parses token by token, but composing 5000 nested sequences exceeds the recursion limit.
    ("[" * 5000 + "]" * 5000 + "\n", "RecursionError"),
    # G5-05: PyYAML keeps the last of duplicate keys, so either order would attest silently.
    ("fail_on_driver_termination: false\nfail_on_driver_termination: true\n",
     "ConstructorError"),
    ("fail_on_driver_termination: true\nfail_on_driver_termination: false\n",
     "ConstructorError"),
    ("fail_on_driver_termination: true\nnested:\n  a: 1\n  a: 2\n", "ConstructorError"),
])
def test_unreadable_eval_config_blocks(tmp_path, eval_config, error):
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"eval_config": eval_config})
    unit = extract(units, ledger, "pair_a")
    assert unit["status"] == "blocked"
    assert codes(unit)["eval_config_unreadable"] == "blocking"
    assert "driver_termination_not_fatal" not in codes(unit)
    [issue] = [item for item in unit["issues"] if item["code"] == "eval_config_unreadable"]
    assert issue["message"].endswith(f": {error}")
    assert "actual_steps" not in unit["observations"][0]
    assert "requested_steps" not in unit["observations"][0]
    assert unit["horizon"]["fail_on_driver_termination"] is None
    assert unit["horizon"]["steps_reported"] is False


def test_merge_keys_and_distinct_keys_still_load(tmp_path):
    # Control for G5-05: a '<<' merge key overridden by an explicit key is not a duplicate.
    config = ("base: &base\n  fail_on_driver_termination: false\n"
              "eval:\n  <<: *base\n  fail_on_driver_termination: true\n"
              "fail_on_driver_termination: true\n")
    units, ledger = build(tmp_path, ["pair_a"], pair_a={"eval_config": config})
    unit = extract(units, ledger, "pair_a")
    assert "eval_config_unreadable" not in codes(unit)
    assert unit["horizon"]["fail_on_driver_termination"] is True
    assert unit["horizon"]["steps_reported"] is True


def test_horizon_attested_by_the_pinned_preset(tmp_path):
    units, ledger = build(tmp_path, ["pair_a"])
    unit = extract(units, ledger, "pair_a")
    assert unit["horizon"]["fail_on_driver_termination"] is True
    assert unit["horizon"]["steps_reported"] is True


def test_observed_evaluator_sha_is_labelled_unauthenticated(tmp_path):
    # F2-10: the SHA is a caller argument, not bound to run evidence here.
    result, _, _ = batch(tmp_path, ["pair_a", "pair_b_late"], CONVENTIONS)
    assert result["observed_evaluator_sha"] == {
        "value": SUPPORTED_EVALUATOR_SHA, "status": "caller_supplied_not_authenticated"}
