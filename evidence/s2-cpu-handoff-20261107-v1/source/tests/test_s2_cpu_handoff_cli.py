"""Public CLI smoke tests for pure command preparation and ASL inspection."""

import json
import os
import subprocess
import sys
from pathlib import Path

from s2_capture_fixture import wire_capture_request
from s2_launch_fixture import launch_request

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    result = subprocess.run(
        [sys.executable, "-B", "-m", "gate.s2", *map(str, args)], cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
        check=False, timeout=30,
    )
    return result


def test_prepare_launch_cli_is_pure_and_has_exact_ten_job_scope(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch_request()))
    result = run_cli("prepare-launch", path)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["pilot"]["policy_rollouts"] == 10
    assert report["commands_executed"] is False
    assert not report["cloud_launch_authorized"]
    assert list(tmp_path.iterdir()) == [path]


def test_missing_launch_inputs_fail_cli(tmp_path):
    path = tmp_path / "incomplete.json"
    request = launch_request()
    request["scene_ids"] = []
    path.write_text(json.dumps(request))
    result = run_cli("prepare-launch", path)
    assert result.returncode == 2
    assert not json.loads(result.stdout)["prepared"]


def test_capture_cli_uses_real_wire_format_without_score_admission(tmp_path):
    request = wire_capture_request(tmp_path)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    result = run_cli("inspect-capture", path, "--root", tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["capture_inspection_passed"]
    assert not report["score_admitted"]
    assert not report["runtime_qualified"]


def test_capture_cli_fails_when_log_bytes_change(tmp_path):
    request = wire_capture_request(tmp_path)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    (tmp_path / "capture.asl").write_bytes(b"torn")
    result = run_cli("inspect-capture", path, "--root", tmp_path)
    assert result.returncode == 2
    assert not json.loads(result.stdout)["capture_inspection_passed"]
