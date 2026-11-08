"""Exercise public offline commands in a fresh interpreter."""

import json
import os
import subprocess
import sys

from s2_ingestion_fixture import fixture_bundle


def run(*args):
    return subprocess.run(
        [sys.executable, "-B", "-m", "gate.s2", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_matrix_names_active_two_arm_counts():
    result = run("matrix")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["full_sweep"]["scheduled_policy_rollouts"] == 2646
    assert output["harmonizer_pilot"]["scheduled_policy_rollouts"] == 10
    assert output["cloud_launch_authorized"] is False


def test_ingest_reads_exact_bundle_and_blocks_changed_file(tmp_path):
    payload, _, _, _ = fixture_bundle(tmp_path)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    result = run("ingest", path, "--root", tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["admitted"]
    (tmp_path / "trace.json").write_text("{}")
    result = run("ingest", path, "--root", tmp_path)
    assert result.returncode == 2
    assert not json.loads(result.stdout)["admitted"]


def test_schedule_command_builds_pilot_without_launch(tmp_path):
    payload = {
        "study_id": "fixture",
        "phase": "harmonizer_pilot",
        "scenes": [
            {
                "scene_id": f"s{i}",
                "road_id": f"r{i}",
                "stratum": "urban",
                "asset_sha256": str(i) * 64,
            }
            for i in range(5)
        ],
        "policies": {"A": "teacher"},
        "protocol_hashes": {"off": "a" * 64, "on": "b" * 64},
        "replicates": 1,
        "requested_steps": 200,
    }
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload))
    result = run("schedule", path)
    assert result.returncode == 0, result.stderr + result.stdout
    output = json.loads(result.stdout)
    assert len(output["jobs"]) == 10
    assert output["cloud_launch_authorized"] is False
    assert output["source_request"] == payload


def test_duplicate_json_keys_fail_at_cli_boundary(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    result = run("ingest", path, "--root", tmp_path)
    assert result.returncode == 2
    assert "Duplicate" in json.loads(result.stdout)["error"]
