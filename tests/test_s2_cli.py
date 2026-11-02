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
