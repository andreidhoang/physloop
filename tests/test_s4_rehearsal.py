"""The synthetic S4 rehearsal composes every B3 module with literal answers (B3 item 8).

Literals are derived independently of the modules:
- E19 float32 advantage for a group of two with reward gap 10 (1.0 vs -9.0): +-0.9999998
  (lane-r3 section 9; (r - mean) / (std + 1e-6) with std = 5);
- optimizer steps: 2 rollouts x 22 valid steps = 44 rows; mini_batch 1, one iteration,
  no zero-norm minibatch -> 44 per trainer step, x 3 trainer steps = 132 per run;
- the capture ledger: one payload, three attempts, one silent retry (KeyError), two successes;
- the cost forecast total $47.70 (tests/test_s4_forecast.py derivation);
- F4 decisions: 40 roads, reference always fails and candidate never (road difference -1;
  protected bounds (-1, -0.824)) -> improved, not ruled out at delta_min 0.03; the guard has
  identical outputs, upper zero_difference_upper(40, 0.05) = 0.088 < 0.1 -> GO; so every run
  is IMPROVEMENT_SUPPORTED and the "all" aggregate is IMPROVEMENT_SUPPORTED.
- G3 capture to checks (synthetic capture through the real sink): three simulate replies,
  all decoded. Session a (the KeyError retry) has no collision series: E11 blocks
  (collision_metric_missing) and E10 cannot derive the offline view (unverified); b has no
  contact (any 0, front + lateral 0); c has front and lateral contact at the last step
  (any 1, additive 1 + 1 = 2). E14: a's failure cause "KeyError: 'collision_any'" is a
  missing-metric retry; it stays unverified (log-density indicators are not in the reply and
  the deviation convention is unregistered). Trainer step: 2 rows, mini_batch 2, mu 1 ->
  ceil(2/2) = 1 minibatch, grad 0.5 != 0 -> 1 optimizer step (E20); counter 0 -> 1 and
  scheduler epoch 0 -> 1 (E07); only expert.w changes (E03). E04: row 0 bitwise equal,
  row 1 differs by 2**-20 = 8 float32 ULP at 1.25 (spacing 2**-23), within max_ulp 8.
  E01: run b repeats run a's seeds and outputs with new session UUIDs -> repeatable, 3 units.
"""

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import rehearse_s4

ROOT = Path(__file__).resolve().parents[1]


def test_rehearsal_summary_literals(tmp_path):
    summary = rehearse_s4.run_rehearsal(tmp_path / "r")
    assert summary["compile"] == {"status": "passed",
                                  "cosmos_toml_overrides": {"train.seed": 7},
                                  "max_optimizer_steps_per_run": 132}
    assert summary["launch"]["status"] == "passed"
    assert summary["launch"]["route"] == "h1_resolved_toml"
    assert summary["launch"]["toml_train_seed"] == 7
    assert summary["launch"]["argv_tail"] == (
        "execution.resolved_config_path='/data/alpagym-runs/20270115T080000Z-"
        + "0" * 32 + "/resolved_config.yaml'")
    # 30 stages: review B3-04 replaced the unused AlpaSim clone (S1-S5) by the cache stages
    # S1-S2 and added R2_alpasim_tree on AlpaGym's cached checkout
    assert summary["bootstrap"] == {"compiled": True, "stages": 30, "gpu_count": 2,
                                    "secrets_required": ["hf_token", "ngc_api_key"]}
    assert summary["ledger"] == {
        "status": "passed",
        "counts": {"payloads": 1, "attempts": 3, "succeeded": 2, "silent_retries": 1,
                   "dropped_payloads": 0, "unterminated_attempts": 0,
                   "never_started_siblings": 0},
        "failure_causes": {"KeyError": 1}}
    advantages = summary["checks"]["advantages"]
    assert [round(a, 7) for a in advantages] == [0.9999998, -0.9999998]
    assert math.isclose(advantages[0], 10 / 2 / (5 + 1e-6), rel_tol=1e-6)
    assert summary["checks"]["expected_optimizer_steps"] == 44
    assert summary["checks"]["skip_status"] == "passed"
    assert summary["checks"]["diagnostics_status"] == "passed"
    assert summary["capture_checks"] == {
        "simulate": {"status": "passed",
                     "counts": {"calls": 3, "decoded": 3, "reply_unavailable": 0,
                                "undecodable": 0, "raised": 0, "rollouts": 3}},
        "semantics": {"status": "blocked",
                      "e10": {"a": "unverified", "b": "passed", "c": "passed"},
                      "e11": {"a": "blocked", "b": "passed", "c": "passed"},
                      "contact": {"a": [None, None], "b": [0.0, 0.0], "c": [1.0, 2.0]},
                      "e14": {"status": "unverified", "retry_causes": {"missing_metric": 1}}},
        "trainer_records": "passed",
        "steps": {"status": "passed", "e20": {"trainer_0": "passed"}, "e07": ["passed"]},
        "e03": {"status": "passed", "changed": ["expert.w"]},
        "e04": {"status": "passed", "verdict": "within_registered_tolerance", "paired_rows": 2,
                "max_ulp": 8, "trace_integrity": "passed"},
        "e01": {"status": "passed", "verdict": "repeatable", "compared_units": 3},
    }
    assert summary["watchdog"] == {"decision": "terminate", "reasons": ["role_dead:rollout_0"],
                                   "cleanup": "verified_clean", "handoff": "work_failed"}
    assert summary["forecast"] == {"status": "passed", "total_usd": "47.7",
                                   "additional_cash_needed_usd": "0"}
    assert summary["evaluation"] == {"requests_passed": 3, "spec_status": "passed",
                                     "rule": "all",
                                     "registered_run_ids": ["run-a", "run-b", "run-c"]}
    assert summary["decision"] == {
        "status": "passed", "aggregate": "IMPROVEMENT_SUPPORTED",
        "outcomes": dict.fromkeys(["run-a", "run-b", "run-c"], "IMPROVEMENT_SUPPORTED"),
        "guards": ["hard_brake_guard"]}
    # review B3-18: the offline converter caveat is an info issue on a compiled plan
    assert summary["codes"] == {"launch": [], "plan": ["hf_offline_cache_resolution_unverified"],
                                "ledger": [], "forecast": []}
    assert summary["cloud_launch_authorized"] is False


def test_rehearsal_is_deterministic_and_refuses_an_existing_workdir(tmp_path):
    first = rehearse_s4.run_rehearsal(tmp_path / "one")
    second = rehearse_s4.run_rehearsal(tmp_path / "two")
    assert first == second
    with pytest.raises(FileExistsError):
        rehearse_s4.run_rehearsal(tmp_path / "one")
    # the only file effects are the synthetic capture streams and their reply blobs (G3:
    # run a's rollout and trainer streams; run b, the E01 repeat, has a rollout stream only)
    root = tmp_path / "one"
    written = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert [w for w in written if "/blobs/" not in w] == [
        "capture/rollout_0.jsonl", "capture/trainer_0.jsonl", "capture_b/rollout_0.jsonl"]
    blobs = [root / w for w in written if "/blobs/" in w]
    assert len(blobs) == 6  # one SimulationReturn per simulate call, three calls per run
    for path in blobs:  # content-addressed: blobs/<sha[:2]>/<sha>
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert (path.parent.name, path.name) == (digest[:2], digest)


def test_rehearsal_script_entrypoint(tmp_path):
    # Runs the script with the local interpreter only (no network, no external tool).
    done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "rehearse_s4.py"),
                           "--workdir", str(tmp_path / "cli")], capture_output=True,
                          text=True, check=False, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr
    assert '"total_usd":"47.7"' in done.stdout
