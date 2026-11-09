"""S2 synthetic raw-file ingestion; deliberately not an AlpaSim ASL decoder.

The Parquet schema/metric names follow the pinned public writer. Execution trace
v1 is a PhysLoop synthetic fixture contract, NOT the upstream trajectory format.
Hashes verify bytes, not origin; successful ingestion never qualifies a runtime,
authenticates a registration, recomputes the scorer, or authorizes a launch.

Tickets: S2.2/S2.5 synthetic ingestion; R4 C2 (scheduled scene_id must equal the upstream
clipgt_id, and one simulator session backs one attempt). Verification level: L1 for every
function here (pure checks over synthetic fixtures); the scene_id -> clipgt_id relation it
enforces is L2 (pinned AlpaSim source, cited in docs/S2_INGESTION.md), untested on a host.
Never: decodes a real ASL, selects among retries, fills a registration value, or launches.
"""

import math
from collections import defaultdict
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq

from gate.alpasim_adapter import IDENTITY_FIELDS, SUPPORTED_EVALUATOR_SHA, normalize_summary
from gate.s2_identity import identity_hash, read_artifact, read_json_bytes
from gate.s2_ledger import reconcile_attempts
from gate.s2_schedule import build_schedule

EVIDENCE_KIND = "synthetic_source_contract_v1"
RAW_AGGREGATIONS = {
    "progress_rel_to_total": "last",
    "progress_rel": "min",
    "gt_dist_traveled_m": "last",
    "collision_front": "max",
    "collision_lateral": "max",
    "offroad": "max",
    "left_corridor_laterally": "max",
    "dist_to_gt_trajectory": "max",
    "lateral_dist_to_gt_trajectory": "max",
}
PARQUET_COLUMNS = {
    "name",
    "timestamps_us",
    "values",
    "valid",
    "time_aggregation",
    "run_name",
    *IDENTITY_FIELDS,
}


class ExtractionError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _require(condition, code, message):
    if not condition:
        raise ExtractionError(code, message)


def _object(value, keys, code):
    _require(type(value) is dict and set(value) == set(keys), code, f"Expected fields: {keys}")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _trace(data, job, identity):
    trace = read_json_bytes(data)
    _object(
        trace,
        {
            "schema_version",
            "kind",
            "upstream_identity",
            "coordinate_frame",
            "start_timestamp_us",
            "step_interval_us",
            "samples",
        },
        "invalid_trace",
    )
    _require(
        type(trace["schema_version"]) is int
        and trace["schema_version"] == 1
        and trace["kind"] == "synthetic_execution_trace_v1",
        "invalid_trace",
        "Only the explicitly synthetic trace-v1 contract is supported",
    )
    _require(trace["upstream_identity"] == identity, "trace_identity_mismatch", "Trace IDs differ")
    _require(
        trace["coordinate_frame"] == "world_m",
        "unsupported_coordinate_frame",
        "Synthetic pose coordinates must be world metres",
    )
    start, interval, samples = (
        trace[k] for k in ("start_timestamp_us", "step_interval_us", "samples")
    )
    _require(_integer(start) and _integer(interval, 1), "invalid_trace", "Invalid trace clock")
    _require(
        type(samples) is list and len(samples) == job["requested_steps"] + 1,
        "horizon_unverified",
        "Require initial synthetic sample plus every requested step",
    )
    diagnostics = {
        "render_coverage": "complete",
        "black_frame_count": 0,
        "first_black_timestamp": None,
        "pose_nonfinite_count": 0,
        "first_nonfinite_timestamp": None,
        "map_coverage": "complete",
    }
    issues, timestamps = [], []
    for index, sample in enumerate(samples):
        _object(
            sample,
            {
                "step_index",
                "timestamp_us",
                "ego_xyz_m",
                "render_status",
                "black_frame",
                "map_available",
            },
            "invalid_trace_sample",
        )
        timestamp = sample["timestamp_us"]
        _require(
            _integer(sample["step_index"])
            and sample["step_index"] == index
            and _integer(timestamp)
            and timestamp == start + index * interval,
            "trace_timestamp_order",
            "Synthetic steps/timestamps must exactly cover schedule",
        )
        timestamps.append(timestamp)
        pose = sample["ego_xyz_m"]
        if type(pose) is not list or len(pose) != 3 or not all(_finite(x) for x in pose):
            diagnostics["pose_nonfinite_count"] += 1
            if diagnostics["first_nonfinite_timestamp"] is None:
                diagnostics["first_nonfinite_timestamp"] = timestamp
            issues.append(("invalid_pose", f"Invalid pose at {timestamp}"))
        if sample["render_status"] != "ok":
            diagnostics["render_coverage"] = "partial"
            issues.append(("render_incomplete", f"Render not observed complete at {timestamp}"))
        if type(sample["black_frame"]) is not bool:
            issues.append(("render_incomplete", f"Black-frame observation unknown at {timestamp}"))
        elif sample["black_frame"]:
            diagnostics["black_frame_count"] += 1
            if diagnostics["first_black_timestamp"] is None:
                diagnostics["first_black_timestamp"] = timestamp
            issues.append(("black_frame_observed", f"Black frame observed at {timestamp}"))
        if sample["map_available"] is not True:
            diagnostics["map_coverage"] = "partial"
            issues.append(("map_unverified", f"Map evidence missing at {timestamp}"))
    return timestamps, diagnostics, issues


def _metrics(data, identity, timestamps):
    try:
        table = pq.read_table(pa.BufferReader(data), use_threads=False)
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise ExtractionError("invalid_parquet", str(exc)) from exc
    _require(
        len(table.column_names) == len(PARQUET_COLUMNS)
        and set(table.column_names) == PARQUET_COLUMNS,
        "invalid_parquet_schema",
        "Columns must match the pinned long-format writer",
    )
    for name in PARQUET_COLUMNS:
        kind = table.schema.field(name).type
        valid = (
            pa.types.is_integer(kind)
            if name == "timestamps_us"
            else pa.types.is_float64(kind)
            if name == "values"
            else pa.types.is_boolean(kind)
            if name == "valid"
            else pa.types.is_string(kind) or pa.types.is_large_string(kind)
        )
        _require(valid, "invalid_parquet_schema", f"Wrong column type: {name}")
    streams, run_names = defaultdict(list), set()
    for row in table.to_pylist():
        _require(
            all(row[k] == identity[k] for k in IDENTITY_FIELDS),
            "raw_identity_mismatch",
            "Raw metrics do not match the launch binding",
        )
        _require(
            type(row["run_name"]) is str and bool(row["run_name"].strip()),
            "raw_identity_mismatch",
            "Missing run_name",
        )
        run_names.add(row["run_name"])
        name, timestamp = row["name"], row["timestamps_us"]
        _require(type(name) is str and bool(name.strip()), "invalid_parquet_schema", "Missing name")
        _require(_integer(timestamp), "metric_timestamp_order", "Invalid metric timestamp")
        _require(
            not streams[name] or timestamp > streams[name][-1]["timestamps_us"],
            "metric_timestamp_order",
            f"Duplicate or nonmonotonic stream: {name}",
        )
        streams[name].append(row)
    _require(len(run_names) == 1, "raw_identity_mismatch", "Mixed/empty run_name in metrics")
    for name, aggregation in RAW_AGGREGATIONS.items():
        _require(name in streams, "missing_raw_metric", f"Required raw stream absent: {name}")
        stream = streams[name]
        _require(
            [row["timestamps_us"] for row in stream] == timestamps,
            "metric_coverage",
            f"Incomplete full synthetic timestamp coverage: {name}",
        )
        _require(
            all(row["time_aggregation"] == aggregation for row in stream),
            "metric_aggregation",
            f"Unexpected aggregation for {name}",
        )
        _require(
            all(row["valid"] is True and _finite(row["values"]) for row in stream),
            "invalid_metric_sample",
            f"Missing, invalid or nonfinite sample in {name}",
        )
        if name in {"collision_front", "collision_lateral", "offroad", "left_corridor_laterally"}:
            _require(
                all(row["values"] in (0.0, 1.0) for row in stream),
                "invalid_metric_sample",
                f"Nonbinary event in {name}",
            )
        if name in {"gt_dist_traveled_m", "dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory"}:
            _require(
                all(row["values"] >= 0 for row in stream),
                "invalid_metric_sample",
                f"Negative distance in {name}",
            )
    deviations = [row["values"] for row in streams["dist_to_gt_trajectory"]]
    _require(all(x >= 0 for x in deviations), "invalid_metric_sample", "Negative deviation")
    return {
        "full_max_deviation_m": max(deviations),
        "metric_names": sorted(streams),
        "run_name": next(iter(run_names)),
        "raw_metric_rows": table.num_rows,
    }


def _binding_issues(attempts, jobs):
    """R4 C2 scene/clip and session binding over every ledger attempt (L1; pinned source L2).

    The wizard's ``scenes.scene_ids`` values become the runtime ``scene_id``, which the pinned
    AlpaSim writer records as ``clipgt_id`` (``docs/S2_INGESTION.md``), so a scheduled job's
    ``scene_id`` must equal its attempt's ``clipgt_id`` (``scene_clip_mismatch``). One simulator
    session ``(clipgt_id, rollout_id)`` may back only one attempt of one job (``session_reused``);
    distinct ``run_uuid`` values cannot disambiguate it. Same semantics as
    ``gate.s2_capture`` (``scene_identity_mismatch`` / ``reused_asl_identity``). Returns
    ``(code, path, message)`` triples in ledger order; both codes block admission.
    """
    found, owners = [], {}
    for attempt in attempts:
        identity, attempt_id = attempt.get("upstream_identity"), attempt.get("attempt_id")
        if identity is None:
            continue
        clip, session = identity["clipgt_id"], (identity["clipgt_id"], identity["rollout_id"])
        job = jobs.get(attempt.get("job_id"))
        if job is not None and job["scene_id"] != clip:
            message = f"Scheduled scene_id {job['scene_id']!r} differs from upstream clipgt_id"
            found.append(("scene_clip_mismatch", attempt_id, f"{message} {clip!r}"))
        owner = owners.setdefault(session, (attempt.get("job_id"), attempt_id))
        if owner != (attempt.get("job_id"), attempt_id):
            message = f"Simulator session (clipgt_id, rollout_id) = {session!r} is already"
            claimed = f"claimed by job {owner[0]!r} attempt {owner[1]!r}"
            found.append(("session_reused", attempt_id, f"{message} {claimed}"))
    return found


def extract_bundle(root, request):
    """Verify a synthetic bundle, preserve attempts, normalize all groups or none.

    A pilot/full schedule requires its complete builder request and exact replay.
    Small synthetic schedules are engineering fixtures, not scientific matrices.
    No retry selection rule is invented: all retries remain blocked by ledger v1.
    """
    report = {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "admitted": False,
        "runtime_qualified": False,
        "cloud_launch_authorized": False,
        "publication_authorized": False,
        "summary_recomputed": False,
        "issues": [],
        "ledger": None,
        "observations": [],
        "batches": [],
        "source_receipts": [],
        "validation_scope": "synthetic raw-file contracts only",
    }

    def issue(code, path, message):
        report["issues"].append({"code": code, "path": path, "message": message})

    def artifact(receipt):
        try:
            data = read_artifact(root, receipt)
        except (ValueError, OSError) as exc:
            raise ExtractionError("artifact_integrity", str(exc)) from exc
        report["source_receipts"].append(deepcopy(receipt))
        return data

    try:
        _object(
            request,
            {
                "schema_version",
                "evidence_kind",
                "evaluator_sha",
                "schedule",
                "schedule_request",
                "events",
                "batches",
                "artifacts",
            },
            "invalid_request",
        )
        _require(
            type(request["schema_version"]) is int and request["schema_version"] == 1,
            "invalid_request",
            "Unsupported ingestion schema",
        )
        _require(
            request["evidence_kind"] == EVIDENCE_KIND,
            "unsupported_evidence_kind",
            "Real ASL/runtime capture is not implemented by this synthetic reader",
        )
        _require(
            request["evaluator_sha"] == SUPPORTED_EVALUATOR_SHA,
            "unsupported_evaluator",
            "Re-audit parser for a different source pin",
        )
        report["request_sha256"] = identity_hash(request)
        schedule = request["schedule"]
        if request["schedule_request"] is not None:
            _require(
                type(request["schedule_request"]) is dict,
                "invalid_schedule_request",
                "Builder request must be an object",
            )
            try:
                generated = build_schedule(**request["schedule_request"])
            except (ValueError, TypeError) as exc:
                raise ExtractionError("invalid_schedule_request", str(exc)) from exc
            _require(
                generated == schedule, "schedule_mismatch", "Schedule differs from builder request"
            )
        else:
            _require(
                type(schedule) is list
                and bool(schedule)
                and all(type(j) is dict and j.get("phase") == "synthetic" for j in schedule),
                "missing_schedule_request",
                "Pilot/full schedules require reproducible source input",
            )
        try:
            events = read_json_bytes(artifact(request["events"]))
        except ExtractionError:
            raise
        except ValueError as exc:
            raise ExtractionError("invalid_events", str(exc)) from exc
        ledger = reconcile_attempts(schedule, events)
        report["ledger"] = ledger
        report["issues"].extend(deepcopy(ledger["issues"]))
        # Do not interpret malformed receipt identities beyond the ledger boundary.
        if not ledger["admitted"]:
            return report
        jobs = {job["job_id"]: job for job in schedule}
        attempts = {attempt["attempt_id"]: attempt for attempt in ledger["attempts"]}
        _require(type(request["artifacts"]) is list, "invalid_artifacts", "Expected artifact array")
        artifacts = {}
        for entry in request["artifacts"]:
            _object(entry, {"attempt_id", "metrics", "trace", "completion"}, "invalid_artifacts")
            attempt_id = entry["attempt_id"]
            _require(
                type(attempt_id) is str and attempt_id in attempts,
                "unexpected_artifact_attempt",
                "Artifact has no launch receipt",
            )
            _require(attempt_id not in artifacts, "duplicate_artifact_attempt", attempt_id)
            artifacts[attempt_id] = entry
        _require(
            set(artifacts) == set(attempts),
            "missing_attempt_artifacts",
            "Every completed attempt needs exact raw artifacts",
        )
        for code, path, message in _binding_issues(ledger["attempts"], jobs):
            issue(code, path, message)
        groups = defaultdict(lambda: {"identities": [], "observations": [], "run_names": {}})
        for attempt_id, attempt in attempts.items():
            job, entry = jobs[attempt["job_id"]], artifacts[attempt_id]
            identity = attempt["upstream_identity"]
            try:
                marker = artifact(entry["completion"])
                _require(marker == b"", "invalid_completion_marker", "Expected empty touch marker")
                timestamps, diagnostics, trace_issues = _trace(
                    artifact(entry["trace"]), job, identity
                )
                for code, message in trace_issues:
                    issue(code, attempt_id, message)
                metrics = _metrics(artifact(entry["metrics"]), identity, timestamps)
                observation = {
                    **identity,
                    "attempt_id": attempt_id,
                    "job_id": job["job_id"],
                    "requested_steps": job["requested_steps"],
                    "actual_steps": len(timestamps) - 1,
                    "completion_validated": not trace_issues,
                    "required_metrics_verified": not trace_issues,
                    "provenance_ref": identity_hash(entry),
                    **metrics,
                    **diagnostics,
                }
                report["observations"].append(observation)
                group = groups[(job["arm"], job["protocol_hash"], job["policy_id"])]
                group["observations"].append(observation)
                group["identities"].append(
                    {
                        **identity,
                        **{
                            k: job[k]
                            for k in ("scene_id", "road_id", "stratum", "replicate_id", "policy_id")
                        },
                    }
                )
                group["run_names"][tuple(identity[k] for k in IDENTITY_FIELDS)] = metrics[
                    "run_name"
                ]
            except (ValueError, OSError) as exc:
                issue(getattr(exc, "code", "invalid_raw_artifact"), attempt_id, str(exc))
        _require(type(request["batches"]) is list, "invalid_batches", "Expected batches array")
        expected = {(j["arm"], j["protocol_hash"]) for j in schedule}
        seen = set()
        for entry in request["batches"]:
            _object(entry, {"arm", "protocol_hash", "summary"}, "invalid_batches")
            _require(
                type(entry["arm"]) is str and type(entry["protocol_hash"]) is str,
                "invalid_batches",
                "Batch keys must be strings",
            )
            key = (entry["arm"], entry["protocol_hash"])
            _require(
                key in expected and key not in seen, "unexpected_batch", "Unmapped/duplicate batch"
            )
            seen.add(key)
            policies = {j["policy_id"] for j in schedule if (j["arm"], j["protocol_hash"]) == key}
            _require(len(policies) == 1, "batch_policy_mismatch", "One policy per arm/protocol")
            policy = next(iter(policies))
            group = groups[(*key, policy)]
            summary = read_json_bytes(artifact(entry["summary"]))
            if type(summary) is dict and type(summary.get("rollouts")) is list:
                for row in summary["rollouts"]:
                    if type(row) is dict and all(type(row.get(k)) is str for k in IDENTITY_FIELDS):
                        row_key = tuple(row[k] for k in IDENTITY_FIELDS)
                        if (
                            row_key in group["run_names"]
                            and row.get("run_name") != group["run_names"][row_key]
                        ):
                            issue(
                                "raw_identity_mismatch",
                                str(key),
                                "Summary run_name differs from raw",
                            )
            normalization = normalize_summary(
                summary,
                group["identities"],
                group["observations"],
                arm=key[0],
                policy_id=policy,
                expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
                observed_evaluator_sha=request["evaluator_sha"],
            )
            for item in normalization["issues"]:
                issue(item["code"], str(key), item["message"])
            report["batches"].append(
                {
                    "arm": key[0],
                    "protocol_hash": key[1],
                    "policy_id": policy,
                    "normalization": normalization,
                }
            )
        _require(seen == expected, "missing_batch", "Every arm/protocol needs a summary")
        report["admitted"] = not report["issues"]
    except (ValueError, TypeError, OSError, KeyError) as exc:
        issue(getattr(exc, "code", "invalid_request"), "request", str(exc))
    if not report["admitted"]:
        for batch in report["batches"]:
            batch["normalization"]["records"] = []
            batch["normalization"]["admitted"] = False
    return report
