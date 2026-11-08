"""Ten invented pilot jobs using the existing synthetic ingestion fixture.

This is test/rehearsal input, not an AlpaSim capture or scientific registration.
The five invented scenes and off/on protocols preserve the adopted pilot shape.
"""

from pathlib import Path

from s2_ingestion_fixture import (
    SHA,
    fixture_bundle,
    json_receipt,
    metric_rows,
    parquet_receipt,
)

from gate.s2_schedule import build_schedule


def build_pilot_fixture(root):
    """Populate an existing empty real directory and return its ingestion request.

    Also retain ``request.json`` for the existing ``gate.s2 ingest`` CLI. Every
    job has three steps, a score of 0.8 and full deviation of 4.2 metres. The
    fixture intentionally retains the original late-deviation negative control.
    """
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Fixture root must not contain symlink components")
    if not root.is_dir():
        raise NotADirectoryError(root)
    if any(root.iterdir()):
        raise FileExistsError("Fixture root must be empty")

    source_request = {
        "study_id": "synthetic-pilot-rehearsal",
        "phase": "harmonizer_pilot",
        "scenes": [
            {
                "scene_id": f"synthetic-scene-{index}",
                "road_id": f"synthetic-road-{index}",
                "stratum": "synthetic-urban",
                "asset_sha256": str(index) * 64,
            }
            for index in range(5)
        ],
        "policies": {"A": "synthetic-teacher"},
        "protocol_hashes": {"off": "a" * 64, "on": "b" * 64},
        "replicates": 1,
        "requested_steps": 3,
    }
    schedule = build_schedule(**source_request)
    events, artifacts = [], []
    summaries = {
        protocol: {"schema_version": 1, "scene_score_enabled": True, "rollouts": []}
        for protocol in source_request["protocol_hashes"].values()
    }
    for index, job in enumerate(schedule):
        directory = root / f"job-{index:02d}"
        directory.mkdir()
        base, job_events, trace, summary = fixture_bundle(directory)
        attempt_id = f"synthetic-attempt-{index:02d}"
        upstream = {
            "run_uuid": f"synthetic-run-{index:02d}",
            "clipgt_id": job["scene_id"],
            "rollout_id": f"synthetic-rollout-{index:02d}",
        }
        for offset, event in enumerate(job_events):
            event.update(
                event_id=f"synthetic-event-{2 * index + offset:02d}",
                job_id=job["job_id"],
                attempt_id=attempt_id,
                timestamp_us=2 * index + offset + 1,
                upstream_identity=upstream,
            )
        events.extend(job_events)
        json_receipt(directory, "events.json", job_events)
        trace["upstream_identity"] = upstream
        summary["rollouts"][0].update(upstream)
        summary["rollouts"][0]["metrics"].update(upstream)
        json_receipt(directory, "summary.json", summary)
        summaries[job["protocol_hash"]]["rollouts"].extend(summary["rollouts"])
        rows = metric_rows()
        for row in rows:
            row.update(upstream)
        entry = {
            "attempt_id": attempt_id,
            "metrics": parquet_receipt(directory, rows),
            "trace": json_receipt(directory, "trace.json", trace),
            "completion": base["artifacts"][0]["completion"],
        }
        for name in ("metrics", "trace", "completion"):
            entry[name] = {**entry[name], "path": f"{directory.name}/{entry[name]['path']}"}
        artifacts.append(entry)

    request = {
        "schema_version": 1,
        "evidence_kind": "synthetic_source_contract_v1",
        "evaluator_sha": SHA,
        "schedule": schedule,
        "schedule_request": source_request,
        "events": json_receipt(root, "events.json", events),
        "batches": [
            {
                "arm": "A",
                "protocol_hash": protocol,
                "summary": json_receipt(root, f"summary-{mode}.json", summaries[protocol]),
            }
            for mode, protocol in source_request["protocol_hashes"].items()
        ],
        "artifacts": artifacts,
    }
    json_receipt(root, "request.json", request)
    return request
