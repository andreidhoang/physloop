"""Descriptive inspection of actual pinned-writer outputs of one S2 unit; diagnostics only.

Tickets: build-spec B1 (lane R1 P5 "descriptive actual-capture inspection": metrics parquet
stream coverage, validity masks with invalid samples preserved, failed summary rows with
``run_uuid`` null, rollout folders without ``_complete``, requested vs effective steps;
runbook section 4 step 6; lane R1 contradictions D3/D5). Agent-authored under
D-018/D-024; serves S2.1-S2.7 [C] (G1, G3-R1, G7).

``inspect_actual_unit(root, receipts, binding, scope=...)`` reads only files named in
``receipts`` (exactly ``{path, sha256, size_bytes}``, e.g. a ``scripts/s2_guest_manifest``
file list of the unit's RUN_DIR or its exported copy) through
``gate.s2_identity.read_artifact`` (no-follow, hash-checked bytes), and describes:

- ``rollouts``: every ``rollouts/<clipgt_id>/<rollout_uuid>/`` folder
  (``gate.s2_capture.scan_rollout_files``) with ``_complete``/``metrics.parquet``/
  ``rollout.asl`` presence and, where present, a metrics inspection: the long-format
  columns of the pinned writer (``gate.s2_extract.PARQUET_COLUMNS``; data.py at AlpaSim
  e121e591), observed Arrow types, identity values against the folder and the bound
  ``run_uuid``, and per metric stream: rows, ``valid`` true/false/null counts (invalid
  samples are preserved by the writer and counted, never dropped or zeroed), null and
  nonfinite values, timestamp range, distinct and duplicate timestamps, strict
  monotonicity and ``time_aggregation`` values; the required raw streams
  (``gate.s2_extract.RAW_AGGREGATIONS``) present/missing, aggregation mismatches and each
  stream's timestamp coverage of the union of required-stream timestamps.
- ``folders_without_complete``: failed or interrupted attempts (lane R2 section 2: the
  marker is written only after a successful evaluation).
- ``summary``: ``aggregate/results-summary.json`` row counts by status, rows whose
  ``run_uuid`` is null (runtime-failure rows, failed_rollouts.py:44-56), rows without a
  folder and folders without a row. Scores are never read into the report.
- ``steps``: the requested horizon (binding), ``simulation_config.n_sim_steps`` of
  ``generated-user-config-0.yaml`` (``yaml.safe_load``) and the effective steps each
  rollout logged in ``Session STARTING ... steps=`` (``gate.s2_capture.parse_runtime_log``
  over ``physloop-capture/compose-up.stdout.txt`` and ``compose-logs.txt``). A difference
  is reported, never resolved: clipped-horizon acceptance is Huy's.
- ``run_uuid_binding``: ``gate.s2_capture.bind_run_uuid`` over ``run_metadata.yaml``.

``status`` is ``blocked`` on an integrity or parse failure (including summary rows whose
``rollout_id`` is not text), ``not_run`` when no receipted input was read at all,
``unverified`` when any required input is absent (``run_metadata.yaml``, the summary, at
least one Compose log, the generated config: issues ``run_metadata_absent``,
``summary_absent``, ``logs_absent``, ``generated_config_absent``), and ``passed`` only when
all of them were read and parsed with their receipts; absence is never a pass. A unit
without rollout folders is reported (``no_rollout_folders``) but is a describable outcome
(e.g. a pre-UUID crash), not missing input. It is inspection completeness only:
``diagnostic_only`` is true, ``score_admitted`` is false and nothing here is an admitted
score, a validity verdict or a horizon decision.

Agent technical defaults (D-018), provisional until the first measured unit:
``MAX_PARQUET_BYTES`` = 256 MiB and ``MAX_TEXT_BYTES`` = 256 MiB per file (one parquet is
read fully into memory by pyarrow).

Verification level: ``inspect_metrics_parquet`` and ``inspect_actual_unit`` L1 over
synthetic files written in the pinned writer's column layout (known answers in
``tests/test_s2_actual.py``, fixtures in ``tests/fixtures/s2_actual``); L2 for the column
layout, log formats and summary row shapes (pinned AlpaSim e121e591 sources vendored
under ``tests/fixtures``); real outputs are L3 (``qualify_on_host``).

This module never executes, writes files, follows symlinks, reads outside receipted
files, fills a missing sample, admits a score or chooses an owner value.
"""

import hashlib
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gate.s2_capture import bind_run_uuid, parse_runtime_log, scan_rollout_files
from gate.s2_extract import PARQUET_COLUMNS, RAW_AGGREGATIONS
from gate.s2_identity import identity_hash, read_artifact, read_json_bytes

ARTIFACT_KIND = "s2_actual_capture_inspection"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
MAX_PARQUET_BYTES = 256 * 1024 * 1024  # agent technical default (D-018)
MAX_TEXT_BYTES = 256 * 1024 * 1024  # agent technical default (D-018)
LOG_PATHS = (("physloop-capture/compose-up.stdout.txt", "compose_up_stdout"),
             ("physloop-capture/compose-logs.txt", "compose_logs"))
SUMMARY_PATH = "aggregate/results-summary.json"
GENERATED_CONFIG_PATH = "generated-user-config-0.yaml"
RUN_METADATA_PATH = "run_metadata.yaml"
# Everything a SafeLoader load can raise on hostile bytes: YAMLError, constructor value
# errors (ValueError, TypeError, OverflowError: "when: 2026-13-45") and RecursionError.
_YAML_LOAD_ERRORS = (yaml.YAMLError, ValueError, TypeError, OverflowError, RecursionError)
_BINDING_FIELDS = {"scene_id", "requested_steps"}
_STRING_COLUMNS = PARQUET_COLUMNS - {"timestamps_us", "values", "valid"}


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(issues, code, path, message, severity="blocking"):
    issues.append({"code": code, "path": path, "severity": severity, "evidence_refs": [path],
                   "message": message})


def _type_ok(column, kind):
    if column == "timestamps_us":
        return pa.types.is_integer(kind)
    if column == "values":
        return pa.types.is_floating(kind)
    if column == "valid":
        return pa.types.is_boolean(kind)
    string_view = getattr(pa.types, "is_string_view", lambda _kind: False)
    return pa.types.is_string(kind) or pa.types.is_large_string(kind) or string_view(kind)


def _sorted_values(values):
    present = sorted({value for value in values if value is not None}, key=str)
    return present + ([None] if None in values else [])


def inspect_metrics_parquet(data, *, clipgt_id, rollout_uuid, run_uuid, path="metrics.parquet"):
    """Describe one metrics.parquet; issues are descriptive except unreadable bytes (L1)."""
    issues = []
    result = {"rows": None, "columns": None, "column_types": None, "schema_matches": False,
              "identity": None, "identity_matches": None, "streams": {},
              "required_streams": None, "invalid_samples_preserved": None, "issues": issues}
    try:
        table = pq.read_table(pa.BufferReader(data), use_threads=False)
    except (pa.ArrowException, OSError, ValueError) as exc:
        _issue(issues, "metrics_unreadable", path, f"Parquet unreadable: {type(exc).__name__}")
        return result
    columns = list(table.column_names)
    types = {name: str(table.schema.field(name).type) for name in columns}
    result.update(rows=table.num_rows, columns=columns, column_types=types)
    missing = sorted(PARQUET_COLUMNS - set(columns))
    extra = sorted(set(columns) - PARQUET_COLUMNS)
    mistyped = sorted(name for name in PARQUET_COLUMNS & set(columns)
                      if not _type_ok(name, table.schema.field(name).type))
    result["schema_matches"] = not (missing or extra or mistyped)
    if not result["schema_matches"]:
        _issue(issues, "metrics_schema_differs", path,
               f"Pinned writer columns differ: missing {missing}, extra {extra}, "
               f"unexpected types {mistyped}", "warning")
    if missing:
        return result
    rows = table.select(sorted(PARQUET_COLUMNS)).to_pylist()
    identity = {name: _sorted_values([row[name] for row in rows])
                for name in ("clipgt_id", "rollout_id", "run_uuid", "run_name")}
    result["identity"] = identity
    matches = {"clipgt_id": identity["clipgt_id"] == [clipgt_id],
               "rollout_id": identity["rollout_id"] == [rollout_uuid],
               "run_uuid": run_uuid is not None and identity["run_uuid"] == [run_uuid]}
    result["identity_matches"] = matches
    if not all(matches.values()):
        _issue(issues, "metrics_identity_mismatch", path,
               f"Identity values differ from folder/run binding: "
               f"{sorted(k for k, ok in matches.items() if not ok)}", "warning")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    invalid_total = 0
    for name in sorted(grouped, key=str):
        stream = grouped[name]
        stamps = [row["timestamps_us"] for row in stream]
        integer_stamps = [stamp for stamp in stamps if type(stamp) is int]
        valid = [row["valid"] for row in stream]
        values = [row["values"] for row in stream]
        nonfinite = sum(1 for value in values
                        if type(value) is float and not math.isfinite(value))
        distinct = set(integer_stamps)
        summary = {
            "rows": len(stream),
            "valid_true": sum(1 for item in valid if item is True),
            "valid_false": sum(1 for item in valid if item is False),
            "valid_null": sum(1 for item in valid if item is None),
            "values_null": sum(1 for value in values if value is None),
            "values_nonfinite": nonfinite,
            "timestamps_null": len(stamps) - len(integer_stamps),
            "first_timestamp_us": min(integer_stamps) if integer_stamps else None,
            "last_timestamp_us": max(integer_stamps) if integer_stamps else None,
            "distinct_timestamps": len(distinct),
            "duplicate_timestamps": len(integer_stamps) - len(distinct),
            "strictly_increasing": len(integer_stamps) == len(stamps) and all(
                a < b for a, b in pairwise(integer_stamps)),
            "time_aggregations": _sorted_values([row["time_aggregation"] for row in stream]),
        }
        invalid_total += summary["valid_false"]
        result["streams"][str(name)] = summary
    result["invalid_samples_preserved"] = invalid_total
    if invalid_total:
        _issue(issues, "invalid_samples_preserved", path,
               f"{invalid_total} samples carry valid=False and are kept as written", "info")
    present = [name for name in RAW_AGGREGATIONS if name in grouped]
    union = set()
    for name in present:
        union |= {row["timestamps_us"] for row in grouped[name]
                  if type(row["timestamps_us"]) is int}
    coverage = {}
    for name in present:
        own = {row["timestamps_us"] for row in grouped[name]}
        coverage[name] = {"covered": len(own & union), "union": len(union)}
    mismatched = sorted(name for name in present
                        if result["streams"][name]["time_aggregations"]
                        != [RAW_AGGREGATIONS[name]])
    result["required_streams"] = {
        "present": sorted(present),
        "missing": sorted(set(RAW_AGGREGATIONS) - set(present)),
        "aggregation_mismatch": mismatched,
        "coverage_of_required_union": coverage,
    }
    if result["required_streams"]["missing"]:
        _issue(issues, "required_streams_missing", path,
               f"Required raw streams absent: {result['required_streams']['missing']}",
               "warning")
    partial = sorted(name for name, item in coverage.items() if item["covered"] < item["union"])
    if partial:
        _issue(issues, "required_stream_coverage_partial", path,
               f"Streams not covering every required timestamp: {partial}", "warning")
    if mismatched:
        _issue(issues, "required_stream_aggregation_differs", path,
               f"Streams with unexpected time_aggregation: {mismatched}", "warning")
    return result


def _read(root, receipts, path, maximum, issues):
    receipt = receipts.get(path)
    if receipt is None:
        return None
    if receipt["size_bytes"] > maximum:
        _issue(issues, "file_exceeds_bound", path,
               f"{receipt['size_bytes']} bytes exceed the {maximum}-byte inspection bound")
        return None
    try:
        return read_artifact(root, receipt)
    except (OSError, ValueError) as exc:
        _issue(issues, "receipt_mismatch", path,
               f"Receipted bytes could not be verified: {type(exc).__name__}: {exc}")
        return None


def _check_inputs(receipts, binding, scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; no default is chosen")
    if (type(binding) is not dict or set(binding) != _BINDING_FIELDS
            or type(binding["scene_id"]) is not str or not binding["scene_id"]
            or type(binding["requested_steps"]) is not int or binding["requested_steps"] <= 0):
        raise ValueError(f"binding requires exactly {sorted(_BINDING_FIELDS)}")
    if type(receipts) is not list or not all(
            type(item) is dict and set(item) == {"path", "sha256", "size_bytes"}
            and type(item["path"]) is str and type(item["size_bytes"]) is int
            for item in receipts):
        raise ValueError("receipts must be a list of {path, sha256, size_bytes}")
    paths = [item["path"] for item in receipts]
    if len(set(paths)) != len(paths):
        raise ValueError("receipt paths must be unique")


def inspect_actual_unit(root, receipts, binding, *, scope,
                        max_parquet_bytes=MAX_PARQUET_BYTES, max_text_bytes=MAX_TEXT_BYTES):
    """Describe one unit's receipted outputs; never an admitted score (L1)."""
    _check_inputs(receipts, binding, scope)
    issues = []
    by_path = {item["path"]: item for item in receipts}
    report = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "gate.s2_actual", "function": "inspect_actual_unit",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {"receipts": identity_hash(receipts), "binding": identity_hash(binding)},
        "units": {"timestamps": "microseconds", "size_bytes": "bytes"},
        "bounds": {"max_parquet_bytes": max_parquet_bytes, "max_text_bytes": max_text_bytes},
        "status": "blocked", "issues": issues, "diagnostic_only": True,
        "score_admitted": False, "admission": "not_applicable",
        "run_uuid_binding": None, "steps": None, "rollouts": [],
        "folders_without_complete": [], "summary": None,
    }
    data = _read(root, by_path, RUN_METADATA_PATH, max_text_bytes, issues)
    if data is not None:
        report["run_uuid_binding"] = bind_run_uuid(data)
        for item in report["run_uuid_binding"]["issues"]:
            _issue(issues, item["code"], RUN_METADATA_PATH, item["message"], "warning")
    elif RUN_METADATA_PATH not in by_path:
        _issue(issues, "run_metadata_absent", RUN_METADATA_PATH, "No run_metadata.yaml",
               "warning")
    read = {"run_metadata": data is not None}
    run_uuid = (report["run_uuid_binding"] or {}).get("run_uuid")

    generated_steps = None
    data = _read(root, by_path, GENERATED_CONFIG_PATH, max_text_bytes, issues)
    read["generated_config"] = data is not None
    if data is None and GENERATED_CONFIG_PATH not in by_path:
        _issue(issues, "generated_config_absent", GENERATED_CONFIG_PATH,
               "No generated-user-config-0.yaml", "warning")
    if data is not None:
        try:
            generated = yaml.safe_load(data)
            config = generated.get("simulation_config") if type(generated) is dict else None
            generated_steps = config.get("n_sim_steps") if type(config) is dict else None
        except _YAML_LOAD_ERRORS:
            _issue(issues, "generated_config_unreadable", GENERATED_CONFIG_PATH,
                   "generated-user-config-0.yaml is not YAML")
    logged = defaultdict(set)
    log_sources = []
    for path, source in LOG_PATHS:
        data = _read(root, by_path, path, max_text_bytes, issues)
        if data is None:
            continue
        parsed = parse_runtime_log(data.decode("utf-8", errors="replace"), source=source)
        log_sources.append(source)
        for item in parsed["issues"]:
            _issue(issues, item["code"], path, item["message"], "warning")
        for event in parsed["events"]:
            if event["kind"] == "session_started":
                logged[event["uuid"]].add(event["steps"])
    read["compose_log"] = bool(log_sources)
    if not any(path in by_path for path, _source in LOG_PATHS):
        _issue(issues, "logs_absent", "physloop-capture",
               f"None of {[path for path, _source in LOG_PATHS]} is receipted", "warning")
    requested = binding["requested_steps"]
    effective = {uuid: sorted(values) for uuid, values in sorted(logged.items())}
    report["steps"] = {
        "requested": requested,
        "generated_config_n_sim_steps": generated_steps
        if type(generated_steps) is int and type(generated_steps) is not bool else None,
        "logged_effective_by_rollout": effective, "log_sources": log_sources,
        "differs_from_requested": sorted(uuid for uuid, values in effective.items()
                                         if values != [requested]),
        "acceptance": "owner_decision_not_made_here",
    }
    if report["steps"]["differs_from_requested"]:
        _issue(issues, "effective_steps_differ_from_requested", "steps",
               f"Logged steps differ from requested {requested} for "
               f"{report['steps']['differs_from_requested']}", "warning")
    if generated_steps is not None and generated_steps != requested:
        _issue(issues, "generated_steps_differ_from_requested", GENERATED_CONFIG_PATH,
               f"n_sim_steps {generated_steps!r} != requested {requested}", "warning")

    scan = scan_rollout_files(sorted(by_path))
    for item in scan["issues"]:
        _issue(issues, item["code"], item["path"], item["message"], "warning")
    folder_ids = set()
    for folder in scan["folders"]:
        entry = dict(folder, metrics_inspection=None)
        folder_ids.add(folder["rollout_uuid"])
        if folder["clipgt_id"] != binding["scene_id"]:
            _issue(issues, "folder_scene_differs", f"rollouts/{folder['clipgt_id']}",
                   "Rollout folder clipgt_id differs from the unit scene", "warning")
        if not folder["complete_marker"]:
            report["folders_without_complete"].append(folder["rollout_uuid"])
        if folder["metrics"]:
            path = f"rollouts/{folder['clipgt_id']}/{folder['rollout_uuid']}/metrics.parquet"
            data = _read(root, by_path, path, max_parquet_bytes, issues)
            if data is not None:
                inspection = inspect_metrics_parquet(
                    data, clipgt_id=folder["clipgt_id"], rollout_uuid=folder["rollout_uuid"],
                    run_uuid=run_uuid, path=path)
                issues.extend(inspection["issues"])
                entry["metrics_inspection"] = inspection
        report["rollouts"].append(entry)
    if not scan["folders"]:
        _issue(issues, "no_rollout_folders", "rollouts", "No rollout folder is receipted",
               "warning")
    if report["folders_without_complete"]:
        _issue(issues, "folders_without_complete", "rollouts",
               f"{len(report['folders_without_complete'])} rollout folders lack _complete",
               "info")

    data = _read(root, by_path, SUMMARY_PATH, max_text_bytes, issues)
    read["summary"] = data is not None
    if data is not None:
        try:
            summary = read_json_bytes(data)
        except ValueError:
            summary = None
            _issue(issues, "summary_unreadable", SUMMARY_PATH, "results-summary.json is not JSON")
        rows = summary.get("rollouts") if type(summary) is dict else None
        if summary is not None and type(rows) is not list:
            _issue(issues, "summary_unreadable", SUMMARY_PATH, "Expected a rollouts list")
        elif rows is not None:
            statuses = defaultdict(int)
            null_run = []
            row_ids = set()
            malformed = 0
            for row in rows:
                if type(row) is not dict or type(row.get("rollout_id")) is not str:
                    statuses["malformed"] += 1
                    malformed += 1
                    continue
                statuses[str(row.get("status"))] += 1
                row_ids.add(row.get("rollout_id"))
                if row.get("run_uuid") is None:
                    null_run.append(row.get("rollout_id"))
            report["summary"] = {
                "rows": len(rows), "status_counts": dict(sorted(statuses.items())),
                "failed_rows_null_run_uuid": _sorted_values(null_run),
                "rows_without_folder": _sorted_values(
                    [rid for rid in row_ids if rid not in folder_ids]),
                "folders_without_row": sorted(folder_ids - row_ids),
            }
            if null_run:
                _issue(issues, "failed_rows_null_run_uuid", SUMMARY_PATH,
                       f"{len(null_run)} summary rows carry run_uuid null", "info")
            if malformed:
                _issue(issues, "summary_rows_malformed", SUMMARY_PATH,
                       f"{malformed} summary rows are not objects with a text rollout_id")
    elif SUMMARY_PATH not in by_path:
        _issue(issues, "summary_absent", SUMMARY_PATH, "No results-summary.json", "warning")
    if any(item["severity"] == "blocking" for item in issues):
        report["status"] = "blocked"
    elif not any(read.values()):
        report["status"] = "not_run"
    elif not all(read.values()):
        report["status"] = "unverified"
    else:
        report["status"] = "passed"
    report["inputs_read"] = dict(sorted(read.items()))
    return report
