"""Invented exact-answer raw files; no licensed or measured simulator data."""

import hashlib
import json

SHA = "e121e59141dc391f6c295513d6dadc840a419149"
IDENTITY = {"run_uuid": "run", "clipgt_id": "clip", "rollout_id": "rollout"}
RAW_METRICS = {
    "progress_rel_to_total": ("last", [0.64] * 4),
    "progress_rel": ("min", [1.0] * 4),
    "gt_dist_traveled_m": ("last", [20.0] * 4),
    "collision_front": ("max", [0.0] * 4),
    "collision_lateral": ("max", [0.0] * 4),
    "offroad": ("max", [0.0] * 4),
    "left_corridor_laterally": ("max", [0.0] * 4),
    "dist_to_gt_trajectory": ("max", [0.0, 0.1, 0.2, 4.2]),
    "lateral_dist_to_gt_trajectory": ("max", [0.0, 0.1, 0.2, 0.3]),
}


def receipt(root, name, data):
    (root / name).write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def json_receipt(root, name, value):
    return receipt(root, name, json.dumps(value, allow_nan=False).encode())


def metric_rows():
    return [
        {
            "name": name,
            "timestamps_us": i * 100_000,
            "values": value,
            "valid": True,
            "time_aggregation": aggregation,
            **IDENTITY,
            "run_name": "bf16",
        }
        for name, (aggregation, values) in RAW_METRICS.items()
        for i, value in enumerate(values)
    ]


def parquet_receipt(root, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return receipt(root, "metrics.parquet", sink.getvalue().to_pybytes())


def fixture_bundle(root):
    job = {
        "job_id": "job",
        "phase": "synthetic",
        "arm": "A",
        "policy_id": "teacher",
        "scene_id": "scene",
        "road_id": "road",
        "stratum": "urban",
        "replicate_id": 0,
        "protocol_hash": "a" * 64,
        "requested_steps": 3,
    }
    events = [
        {
            "event_id": "start",
            "job_id": "job",
            "attempt_id": "attempt",
            "type": "started",
            "timestamp_us": 1,
            "parent_attempt_id": None,
            "retry_reason": None,
            "upstream_identity": IDENTITY,
        },
        {
            "event_id": "end",
            "job_id": "job",
            "attempt_id": "attempt",
            "type": "completed",
            "timestamp_us": 2,
            "reason": "exact_horizon",
            "upstream_identity": IDENTITY,
        },
    ]
    trace = {
        "schema_version": 1,
        "kind": "synthetic_execution_trace_v1",
        "upstream_identity": IDENTITY,
        "coordinate_frame": "world_m",
        "start_timestamp_us": 0,
        "step_interval_us": 100_000,
        "samples": [
            {
                "step_index": i,
                "timestamp_us": i * 100_000,
                "ego_xyz_m": [float(i), 0.0, 0.0],
                "render_status": "ok",
                "black_frame": False,
                "map_available": True,
            }
            for i in range(4)
        ],
    }
    scoring = {
        "progress_clipped_rel": 0.64,
        "progress_rel": 1.0,
        "gt_dist_traveled_m": 20.0,
        "collision_at_fault": 0.0,
        "offroad": 0.0,
        "left_corridor_laterally": 0.0,
        "dist_to_gt_trajectory": 0.2,
        "lateral_dist_to_gt_trajectory": 0.2,
        "progress_score": 0.8,
    }
    summary = {
        "schema_version": 1,
        "scene_score_enabled": True,
        "rollouts": [
            {
                **IDENTITY,
                "run_name": "bf16",
                "status": "pass",
                "passed": True,
                "score": 0.8,
                "failure_reason": None,
                "metrics": {**IDENTITY, "run_name": "bf16", **scoring},
                "score_metrics": scoring,
            }
        ],
    }
    request = {
        "schema_version": 1,
        "evidence_kind": "synthetic_source_contract_v1",
        "evaluator_sha": SHA,
        "schedule": [job],
        "schedule_request": None,
        "events": json_receipt(root, "events.json", events),
        "batches": [
            {
                "arm": "A",
                "protocol_hash": "a" * 64,
                "summary": json_receipt(root, "summary.json", summary),
            }
        ],
        "artifacts": [
            {
                "attempt_id": "attempt",
                "metrics": parquet_receipt(root, metric_rows()),
                "trace": json_receipt(root, "trace.json", trace),
                "completion": receipt(root, "_complete", b""),
            }
        ],
    }
    return request, events, trace, summary
