"""Verdict CLI parses with the strict S2 reader (R4 C3; agent-authored, D-024).

Known answers: 220 roads, every road difference zero, alpha 0.05. The locked upper bound is
1 - 0.025 ** (1 / 220) = 0.01663..., so margin 0.01 is INSUFFICIENT and margin 0.2 would be GO.
The R4 repro bytes carry ``"margin":0.01,"margin":0.2``; a last-wins parser turned the registered
INSUFFICIENT into GO. Margins here are test inputs, not registered values.
"""

import json
import os
import subprocess
import sys

import pytest

from gate.__main__ import main

SPEC_HEAD = (
    '{"reference":"teacher","candidate":"candidate","protocol":{"identity":"p","seed_mode":'
    '"unseeded","config_hash":"c","evaluator":"e","score_definition":"s",'
    '"deviation_convention":"d"},"required_strata":["urban"],"alpha":0.05,'
    '"metric":{"name":"m","kind":"binary","higher_is_worse":true,"unit":"fraction"},'
)
RECORDS = [
    {"arm": arm, "scene_id": f"scene-{i}", "road_id": f"road-{i}", "stratum": "urban",
     "replicate_id": 0, "value": 0, "completed": True, "deviation_m": None, "seed": None}
    for i in range(220)
    for arm in ("A", "B")
]


def request_text(margins):
    return '{"schema_version":1,"spec":' + SPEC_HEAD + margins + '},"records":' + json.dumps(
        RECORDS
    ) + "}"


def run_in_process(tmp_path, capsys, text, name="request.json"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    code = main(["verdict", str(path)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_registered_margin_known_answer_is_insufficient(tmp_path, capsys):
    code, out, _ = run_in_process(tmp_path, capsys, request_text('"margin":0.01'))
    assert code == 0
    result = json.loads(out)
    assert result["comparison"]["margin"] == 0.01
    assert result["verdict"]["overall"] == "INSUFFICIENT"
    upper = result["verdict"]["details"]["urban"]["upper"]
    assert upper == pytest.approx(1 - 0.025 ** (1 / 220), abs=1e-9)


def test_wider_registered_margin_control_is_go(tmp_path, capsys):
    code, out, _ = run_in_process(tmp_path, capsys, request_text('"margin":0.2'))
    assert code == 0 and json.loads(out)["verdict"]["overall"] == "GO"


def test_duplicate_margin_key_exits_2_with_error_json(tmp_path, capsys):
    code, out, err = run_in_process(tmp_path, capsys, request_text('"margin":0.01,"margin":0.2'))
    assert code == 2
    error = json.loads(out)
    assert error["verdict_emitted"] is False
    assert error["error"]["message"] == "Duplicate JSON object key: 'margin'"
    assert "verdict" not in error and "comparison" not in error
    assert "input or execution error" in err


@pytest.mark.parametrize(
    "payload",
    [
        request_text('"margin":NaN'),
        request_text('"margin":Infinity'),
        b'{"schema_version":1,"spec":{},"records":[]}\xff',
        '{"schema_version":1,"schema_version":1}',
    ],
)
def test_nonstandard_json_is_rejected(tmp_path, capsys, payload):
    code, out, _ = run_in_process(tmp_path, capsys, payload)
    assert code == 2
    assert json.loads(out)["verdict_emitted"] is False


def test_missing_file_exits_2_with_error_json(tmp_path, capsys):
    code = main(["verdict", str(tmp_path / "absent.json")])
    out = capsys.readouterr().out
    assert code == 2 and json.loads(out)["error"]["type"] == "FileNotFoundError"


def test_fresh_interpreter_rejects_the_r4_repro_bytes(tmp_path):
    path = tmp_path / "duplicated.json"
    path.write_text(request_text('"margin":0.01,"margin":0.2'))
    result = subprocess.run(
        [sys.executable, "-B", "-m", "gate", "verdict", str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["message"] == "Duplicate JSON object key: 'margin'"
    assert "error" in result.stderr
