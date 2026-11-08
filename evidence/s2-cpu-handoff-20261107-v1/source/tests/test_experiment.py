"""Evidence orchestration tests, not the qualification experiment."""

import hashlib
import json

import pytest

from gate.experiment import configurations, cp_upper, run_experiment


def test_frozen_cells_and_calibration_tail():
    cases = configurations()
    assert len(cases) == 10
    assert len({c["name"] for c in cases}) == 10
    assert all(c["margin"] > 0 for c in cases)
    assert cp_upper(0, 1000, 0.0025) == pytest.approx(1 - 0.0025**0.001)
    assert cp_upper(1000, 1000, 0.0025) == 1


def test_evidence_does_not_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path)


def test_runner_writes_replayable_bundle(tmp_path, monkeypatch):
    # Infrastructure-only check, one deterministic identical case and no power computation.
    import gate.experiment as runner

    original = runner.configurations
    cases = original()
    for case in cases:
        case["generator"] = {**case["generator"], "q_plus": 0.0, "q_minus": 0.0}
        case["generator"].pop("effects", None)
    monkeypatch.setattr(runner, "configurations", lambda: cases)
    monkeypatch.setattr(runner, "power_tables", lambda: {"fixture": [{"value": 1}]})
    path = tmp_path / "run"
    summary = run_experiment(path, trials=1)
    assert summary["status"] == "complete"
    assert summary["qualification"] is False
    assert (path / "source.zip").is_file()
    checksums = json.loads((path / "checksums.json").read_text())
    for name, digest in checksums.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    assert len((path / "trials.jsonl").read_text().splitlines()) == 10


def test_cli_invalid_json(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "bad.json"
    path.write_text("{}")
    result = subprocess.run(
        [sys.executable, "-m", "gate", "verdict", str(path)], capture_output=True, check=False
    )
    assert result.returncode == 2
    assert b"error" in result.stderr


def test_source_snapshot_includes_manifest_and_fixture_dependencies(tmp_path, monkeypatch):
    import gate.experiment as runner

    required = {
        runner.EXPERIMENT,
        "pyproject.toml",
        "uv.lock",
        "gate/example.py",
        "tests/test_example.py",
        "experiments/S2_MANIFEST_DRAFT.json",
        "experiments/S2_HARMONIZER_PILOT_DRAFT.json",
        "experiments/2026-11-02-s2-pilot.md",
        "tests/fixtures/alpasim/provenance.json",
        "tests/fixtures/alpasim/LICENSE",
        "tests/fixtures/alpasim/README.md",
        "tests/fixtures/alpasim/ground_truth_reference.py",
    }
    excluded = {
        "tests/fixtures/alpasim/__pycache__/ignored.py",
        "tests/fixtures/alpasim/__pycache__/ignored.pyc",
        "tests/fixtures/alpasim/ignored.pyc",
        "tests/fixtures/alpasim/ignored.pyo",
    }
    for name in required | excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture dependency", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    captured = {str(path.relative_to(tmp_path)) for path in runner._sources()}
    assert required <= captured
    assert not excluded & captured
