"""CPU-04 known answers for the S2 validity and complete-denominator report.

Inputs are produced by the real W2 consumer (``gate.s2_ledger.reconcile_attempts``)
and, end to end, by the real synthetic W3 reader (``gate.s2_extract.extract_bundle``).
Observation rows copy the exact key set that reader emits. Every literal below is
hand-derived from the readiness specification (CPU-04), not from the implementation.
No test touches a network, a provider CLI or a GPU tool.
"""

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
from s2_ingestion_fixture import fixture_bundle, json_receipt

from gate.s2_diagnostics import (
    DEVIATION_BINS,
    INDICATORS,
    describe_validity,
    deviation_bin,
)
from gate.s2_extract import extract_bundle
from gate.s2_identity import identity_hash
from gate.s2_ledger import reconcile_attempts

FIXTURES = Path(__file__).parent / "fixtures" / "s2_analysis"
PROTOCOL = "a" * 64
OTHER_PROTOCOL = "b" * 64
ENVELOPE_ONLY = {
    "schema_version": 1,
    "definition_id": "synthetic-envelope-only",
    "source_ref": "test",
    "invalid_when": ["out_of_envelope"],
}


def make_job(scene, arm="A", replicate=0, protocol=PROTOCOL, phase="synthetic", **changes):
    return {
        "job_id": f"{phase}-{arm}-{scene}-{replicate}-{protocol[:2]}",
        "phase": phase,
        "arm": arm,
        "policy_id": "teacher" if arm == "A" else "reexport",
        "scene_id": scene,
        "road_id": f"road-{scene}",
        "stratum": "urban",
        "replicate_id": replicate,
        "protocol_hash": protocol,
        "requested_steps": 3,
        **changes,
    }


def identity_of(attempt_id):
    return {"run_uuid": f"run-{attempt_id}", "clipgt_id": "clip", "rollout_id": attempt_id}


class Recorder:
    """Emits ledger events in nondecreasing recorder time."""

    def __init__(self):
        self.events, self.clock = [], 0

    def attempt(self, job, attempt_id, kind="completed", parent=None):
        upstream = identity_of(attempt_id)
        self.clock += 1
        self.events.append(
            {
                "event_id": f"start-{attempt_id}",
                "job_id": job["job_id"],
                "attempt_id": attempt_id,
                "type": "started",
                "timestamp_us": self.clock,
                "parent_attempt_id": parent,
                "retry_reason": None if parent is None else "synthetic restart",
                "upstream_identity": upstream,
            }
        )
        self.clock += 1
        self.events.append(
            {
                "event_id": f"end-{attempt_id}",
                "job_id": job["job_id"],
                "attempt_id": attempt_id,
                "type": kind,
                "timestamp_us": self.clock,
                "reason": "exact_horizon" if kind == "completed" else "synthetic failure",
                "upstream_identity": upstream,
            }
        )


def observation(job, attempt_id, deviation=0.5, **changes):
    """Exactly the key set ``gate.s2_extract.extract_bundle`` emits per observation."""
    return {
        **identity_of(attempt_id),
        "attempt_id": attempt_id,
        "job_id": job["job_id"],
        "requested_steps": job["requested_steps"],
        "actual_steps": job["requested_steps"],
        "completion_validated": True,
        "required_metrics_verified": True,
        "provenance_ref": "0" * 64,
        "full_max_deviation_m": deviation,
        "metric_names": ["dist_to_gt_trajectory"],
        "run_name": "bf16",
        "raw_metric_rows": 4,
        "render_coverage": "complete",
        "black_frame_count": 0,
        "first_black_timestamp": None,
        "pose_nonfinite_count": 0,
        "first_nonfinite_timestamp": None,
        "map_coverage": "complete",
        **changes,
    }


def single_arm(deviations, **observation_changes):
    jobs = [make_job(f"scene-{i}") for i in range(len(deviations))]
    recorder = Recorder()
    observations = []
    for index, (job, deviation) in enumerate(zip(jobs, deviations)):
        recorder.attempt(job, f"att-{index}")
        observations.append(observation(job, f"att-{index}", deviation, **observation_changes))
    return reconcile_attempts(jobs, recorder.events), observations


def paired(pairs):
    jobs, recorder, observations = [], Recorder(), []
    for pair in pairs:
        for arm in ("A", "B"):
            job = make_job(pair["scene_id"], arm=arm)
            attempt_id = f"att-{arm}-{pair['scene_id']}"
            jobs.append(job)
            recorder.attempt(job, attempt_id)
            observations.append(observation(job, attempt_id, pair[arm]))
    return reconcile_attempts(jobs, recorder.events), observations


def codes(report):
    return {item["code"] for item in report["issues"]}


def walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def no_admission_anywhere(report):
    for node in walk(report):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("admitted", "gate_eligible", "admitted_verdict") and value not in (
                    False,
                    None,
                ):
                    return False
    return True


# --- deviation bins ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "[0,1)"),
        (0.999999, "[0,1)"),
        (1, "[1,2)"),
        (1.0, "[1,2)"),
        (1.999999, "[1,2)"),
        (2.0, "[2,4)"),
        (3.999999, "[2,4)"),
        (4.0, "[4,inf)"),
        (4.2, "[4,inf)"),
        (1e9, "[4,inf)"),
        (None, "unknown"),
        (-0.1, "unknown"),
        (True, "unknown"),
        (False, "unknown"),
        ("1.0", "unknown"),
        (float("nan"), "unknown"),
        (float("inf"), "unknown"),
    ],
)
def test_half_open_bins_with_unknown(value, expected):
    assert deviation_bin(value) == expected


def test_literal_five_value_known_answer():
    ledger, observations = single_arm([0.5, 1, 2, 4, None])
    report = describe_validity(ledger, observations)
    assert DEVIATION_BINS == ("[0,1)", "[1,2)", "[2,4)", "[4,inf)")
    assert report["deviation_bins"]["counts"] == {
        "[0,1)": 1,
        "[1,2)": 1,
        "[2,4)": 1,
        "[4,inf)": 1,
        "unknown": 1,
    }
    assert report["deviation_bins"]["denominator"] == 5
    assert "deviation_unknown" in codes(report)
    assert report["out_of_envelope"]["count"] == 1


@pytest.mark.parametrize("bad", [-0.5, True, "2.0"])
def test_invalid_deviation_is_unknown_not_zero(bad):
    ledger, observations = single_arm([bad])
    report = describe_validity(ledger, observations)
    assert report["deviation_bins"]["counts"]["unknown"] == 1
    assert report["deviation_bins"]["counts"]["[0,1)"] == 0
    assert "invalid_deviation" in codes(report)


def test_missing_deviation_key_is_unknown():
    ledger, observations = single_arm([0.5])
    del observations[0]["full_max_deviation_m"]
    report = describe_validity(ledger, observations)
    assert report["deviation_bins"]["counts"]["unknown"] == 1
    assert report["units"][0]["deviation_m"] is None


def test_non_json_input_fails_closed():
    ledger, observations = single_arm([float("nan")])
    with pytest.raises(ValueError):
        describe_validity(ledger, observations)


# --- late deviation, end to end through the real synthetic reader ------------------


def test_late_deviation_beyond_truncated_summary_stays_out_of_envelope(tmp_path):
    request, _, _, summary = fixture_bundle(tmp_path)
    # The summary aggregate stops at 0.2 m; the full stream reaches 4.2 m at the last step.
    assert summary["rollouts"][0]["score_metrics"]["dist_to_gt_trajectory"] == 0.2
    extracted = extract_bundle(tmp_path, request)
    assert extracted["admitted"], extracted["issues"]
    records = extracted["batches"][0]["normalization"]["records"]
    report = describe_validity(extracted["ledger"], extracted["observations"], records)
    assert report["deviation_bins"]["counts"]["[4,inf)"] == 1
    assert report["deviation_bins"]["counts"]["[0,1)"] == 0
    assert report["out_of_envelope"]["count"] == 1
    assert report["out_of_envelope"]["job_ids"] == ["job"]
    assert report["records"]["bound"] == 1
    assert report["records"]["out_of_envelope"] == 1
    assert not {"record_observation_mismatch", "record_unbound"} & codes(report)
    assert report["status"] == "passed"
    assert report["scope"] == "synthetic"
    assert no_admission_anywhere(report)


def test_current_producer_leaves_corridor_and_overshoot_unknown(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    extracted = extract_bundle(tmp_path, request)
    report = describe_validity(extracted["ledger"], extracted["observations"])
    indicators = report["indicators"]
    # The pinned summary says left_corridor_laterally == 0.0; that is not full-stream evidence.
    assert indicators["corridor_departure"] == {"true": 0, "false": 0, "unknown": 1}
    assert indicators["endpoint_overshoot"] == {"true": 0, "false": 0, "unknown": 1}
    assert indicators["terminal_not_exact_horizon"] == {"true": 0, "false": 1, "unknown": 0}
    assert report["terminal"] == {"exact_horizon": 1, "short_horizon": 0, "unknown": 0}
    assert indicators["render_incomplete"] == {"true": 0, "false": 1, "unknown": 0}
    assert indicators["map_incomplete"] == {"true": 0, "false": 1, "unknown": 0}


# --- independent indicators ---------------------------------------------------------


def test_black_frame_and_nonfinite_pose_are_both_counted():
    ledger, observations = single_arm(
        [0.5], black_frame_count=1, first_black_timestamp=100, pose_nonfinite_count=1
    )
    report = describe_validity(ledger, observations)
    assert report["indicators"]["black_frame"] == {"true": 1, "false": 0, "unknown": 0}
    assert report["indicators"]["nonfinite_pose"] == {"true": 1, "false": 0, "unknown": 0}
    unit = report["units"][0]
    assert unit["indicators"]["black_frame"] is True
    assert unit["indicators"]["nonfinite_pose"] is True


def test_zero_black_frames_with_partial_render_is_unknown():
    ledger, observations = single_arm([0.5], render_coverage="partial")
    report = describe_validity(ledger, observations)
    assert report["indicators"]["black_frame"] == {"true": 0, "false": 0, "unknown": 1}
    assert report["indicators"]["render_incomplete"] == {"true": 1, "false": 0, "unknown": 0}


@pytest.mark.parametrize(
    "field,indicator",
    [
        ("map_coverage", "map_incomplete"),
        ("render_coverage", "render_incomplete"),
        ("pose_nonfinite_count", "nonfinite_pose"),
        ("black_frame_count", "black_frame"),
    ],
)
def test_missing_metric_is_unknown_not_false(field, indicator):
    ledger, observations = single_arm([0.5])
    del observations[0][field]
    report = describe_validity(ledger, observations)
    assert report["indicators"][indicator] == {"true": 0, "false": 0, "unknown": 1}
    assert report["units"][0]["indicators"][indicator] is None


def test_none_metric_is_unknown_not_false():
    ledger, observations = single_arm([0.5], map_coverage=None, pose_nonfinite_count=None)
    report = describe_validity(ledger, observations)
    assert report["indicators"]["map_incomplete"]["unknown"] == 1
    assert report["indicators"]["nonfinite_pose"]["unknown"] == 1


def test_full_stream_extension_fields_are_counted():
    ledger, observations = single_arm(
        [0.5, 0.5, 0.5],
    )
    observations[0]["corridor_departure"] = True
    observations[1]["corridor_departure"] = False
    observations[0]["endpoint_overshoot"] = False
    report = describe_validity(ledger, observations)
    assert report["indicators"]["corridor_departure"] == {"true": 1, "false": 1, "unknown": 1}
    assert report["indicators"]["endpoint_overshoot"] == {"true": 0, "false": 1, "unknown": 2}


def test_invalid_extension_value_is_unknown_with_issue():
    ledger, observations = single_arm([0.5])
    observations[0]["corridor_departure"] = 1.0
    report = describe_validity(ledger, observations)
    assert report["indicators"]["corridor_departure"]["unknown"] == 1
    assert "invalid_observation_value" in codes(report)


def test_short_horizon_is_terminal_not_exact():
    ledger, observations = single_arm([0.5], actual_steps=2)
    report = describe_validity(ledger, observations)
    assert report["terminal"] == {"exact_horizon": 0, "short_horizon": 1, "unknown": 0}
    assert report["indicators"]["terminal_not_exact_horizon"]["true"] == 1


def test_unvalidated_completion_leaves_terminal_unknown():
    ledger, observations = single_arm([0.5], completion_validated=False)
    report = describe_validity(ledger, observations)
    assert report["terminal"]["unknown"] == 1


# --- A-valid x B-valid table --------------------------------------------------------


def load_pairs_fixture():
    return json.loads((FIXTURES / "validity_pairs.json").read_text())


def test_two_by_two_literal_known_answer():
    fixture = load_pairs_fixture()
    ledger, observations = paired(fixture["pairs"])
    report = describe_validity(
        ledger, observations, validity_definition=fixture["validity_definition"]
    )
    expected = fixture["expected"]
    table = report["validity_table"]
    assert ledger["counts"]["scheduled_jobs"] == expected["logical_jobs"] == 10
    assert table["status"] == "passed"
    assert table["pairs_total"] == expected["pairs_total"] == 5
    assert table["table"] == expected["table"]
    assert sum(table["table"].values()) == expected["table_total"] == 4
    assert table["outside_table"] == expected["outside_table"]
    assert report["deviation_bins"]["counts"] == expected["deviation_bins"]
    assert table["definition"]["status"] == "supplied_not_authenticated"
    assert table["definition"]["sha256"] == identity_hash(fixture["validity_definition"])
    by_scene = {row["scene_id"]: row["cell"] for row in table["pairs"]}
    assert by_scene == {pair["scene_id"]: pair["expected"] for pair in fixture["pairs"]}
    unknown = [row for row in table["pairs"] if row["cell"] == "unknown"]
    assert unknown[0]["unknown_arms"] == ["B"]


def test_two_by_two_without_definition_is_blocked_but_counts_pairs():
    fixture = load_pairs_fixture()
    ledger, observations = paired(fixture["pairs"])
    report = describe_validity(ledger, observations)
    table = report["validity_table"]
    assert table["status"] == "blocked"
    assert table["code"] == "validity_definition_unregistered"
    assert table["table"] is None
    assert table["pairs_total"] == 5
    assert report["deviation_bins"]["counts"] == fixture["expected"]["deviation_bins"]
    assert all(unit["validity"] is None for unit in report["units"])


def test_known_invalid_criterion_dominates_unknown_criterion():
    fixture = load_pairs_fixture()
    ledger, observations = paired(fixture["pairs"])
    definition = {**ENVELOPE_ONLY, "invalid_when": ["out_of_envelope", "corridor_departure"]}
    report = describe_validity(ledger, observations, validity_definition=definition)
    table = report["validity_table"]
    # Corridor is unknown everywhere: deviation < 4 m -> unknown; deviation >= 4 m -> invalid.
    assert table["table"] == {
        "A_valid_B_valid": 0,
        "A_valid_B_invalid": 0,
        "A_invalid_B_valid": 0,
        "A_invalid_B_invalid": 1,
    }
    assert table["outside_table"] == {"unknown_pairs": 4, "unpaired_jobs": 0}


@pytest.mark.parametrize(
    "definition",
    [
        {**ENVELOPE_ONLY, "invalid_when": []},
        {**ENVELOPE_ONLY, "invalid_when": ["made_up"]},
        {**ENVELOPE_ONLY, "invalid_when": ["out_of_envelope", "out_of_envelope"]},
        {**ENVELOPE_ONLY, "schema_version": 2},
        {**ENVELOPE_ONLY, "definition_id": ""},
        {key: value for key, value in ENVELOPE_ONLY.items() if key != "source_ref"},
        {**ENVELOPE_ONLY, "threshold_m": 3.0},
    ],
)
def test_malformed_definition_blocks_table(definition):
    ledger, observations = paired(load_pairs_fixture()["pairs"])
    report = describe_validity(ledger, observations, validity_definition=definition)
    assert report["validity_table"]["status"] == "blocked"
    assert report["validity_table"]["code"] == "invalid_validity_definition"
    assert report["validity_table"]["table"] is None
    assert report["status"] == "blocked"


def test_pilot_off_on_modes_are_not_arms():
    jobs = [
        make_job(f"scene-{i}", protocol=protocol, phase="harmonizer_pilot")
        for i in range(2)
        for protocol in (PROTOCOL, OTHER_PROTOCOL)
    ]
    recorder, observations = Recorder(), []
    for index, job in enumerate(jobs):
        recorder.attempt(job, f"att-{index}")
        observations.append(observation(job, f"att-{index}"))
    ledger = reconcile_attempts(jobs, recorder.events)
    report = describe_validity(
        ledger, observations, validity_definition=ENVELOPE_ONLY, scope="synthetic"
    )
    assert report["validity_table"]["status"] == "not_applicable"
    assert report["validity_table"]["code"] == "single_arm_schedule"
    assert report["validity_table"]["pairs_total"] == 0


def test_unpaired_arm_is_outside_table():
    fixture = load_pairs_fixture()
    pairs = fixture["pairs"][:4]
    jobs, recorder, observations = [], Recorder(), []
    for pair in pairs:
        for arm in ("A", "B"):
            job = make_job(pair["scene_id"], arm=arm)
            jobs.append(job)
            recorder.attempt(job, f"att-{arm}-{pair['scene_id']}")
            observations.append(observation(job, f"att-{arm}-{pair['scene_id']}", pair[arm]))
    lone = make_job("scene-lonely", arm="A")
    jobs.append(lone)
    recorder.attempt(lone, "att-lone")
    observations.append(observation(lone, "att-lone"))
    report = describe_validity(
        reconcile_attempts(jobs, recorder.events), observations, validity_definition=ENVELOPE_ONLY
    )
    table = report["validity_table"]
    assert sum(table["table"].values()) == 4
    assert table["outside_table"] == {"unknown_pairs": 0, "unpaired_jobs": 1}
    assert table["unpaired_job_ids"] == [lone["job_id"]]


# --- complete denominator, retries and rejected batches -----------------------------


def test_failed_attempt_then_retry_is_one_job_two_attempts_one_retry():
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1", kind="failed")
    recorder.attempt(job, "att-2", parent="att-1")
    ledger = reconcile_attempts([job], recorder.events)
    assert ledger["admitted"] is False
    observations = [observation(job, "att-2", 0.5)]
    report = describe_validity(ledger, observations, validity_definition=ENVELOPE_ONLY)
    counts = report["denominators"]
    assert counts["planned_jobs"] == 1
    assert counts["launched_attempts"] == 2
    assert counts["retries"] == 1
    assert counts["jobs_with_retries"] == 1
    assert counts["failed_attempts"] == 1
    assert counts["completed_attempts"] == 1
    assert counts["selected_observations"] == 0
    assert counts["bound_observations"] == 1
    excluded = {row["job_id"]: row["reasons"] for row in report["excluded_units"]}
    assert "retry_selection_unregistered" in excluded[job["job_id"]]
    assert "batch_not_admitted" in excluded[job["job_id"]]
    # Retry observations stay visible at attempt level, but no attempt is chosen.
    assert report["units"][0]["described_attempt_id"] is None
    assert report["units"][0]["deviation_bin"] == "unknown"
    assert [row["attempt_id"] for row in report["observations"]] == ["att-2"]
    assert report["observations"][0]["deviation_bin"] == "[0,1)"
    assert report["diagnostic_only"] is True
    assert report["gate_eligible"] is False
    assert no_admission_anywhere(report)
    assert report["records"]["supplied"] is False


def test_rejected_batch_still_reports_every_unit_with_reason():
    jobs = [make_job("scene-1"), make_job("scene-2"), make_job("scene-3")]
    recorder = Recorder()
    recorder.attempt(jobs[0], "att-1")
    recorder.attempt(jobs[1], "att-2", kind="cancelled")
    ledger = reconcile_attempts(jobs, recorder.events)
    observations = [observation(jobs[0], "att-1", 4.5)]
    report = describe_validity(ledger, observations)
    excluded = {row["job_id"]: row["reasons"] for row in report["excluded_units"]}
    assert set(excluded) == {job["job_id"] for job in jobs}
    assert excluded[jobs[0]["job_id"]] == ["batch_not_admitted"]
    assert "job_not_completed" in excluded[jobs[1]["job_id"]]
    assert "observation_missing" in excluded[jobs[1]["job_id"]]
    assert "missing_job" in excluded[jobs[2]["job_id"]]
    assert report["denominators"]["jobs_without_attempt"] == 1
    assert report["denominators"]["cancelled_attempts"] == 1
    assert report["deviation_bins"]["counts"]["[4,inf)"] == 1
    assert report["deviation_bins"]["counts"]["unknown"] == 2
    assert report["out_of_envelope"]["job_ids"] == [jobs[0]["job_id"]]
    assert report["ledger_admitted"] is False
    assert no_admission_anywhere(report)


def test_records_from_a_rejected_batch_are_flagged_not_used():
    jobs = [make_job("scene-1"), make_job("scene-2")]
    recorder = Recorder()
    recorder.attempt(jobs[0], "att-1")
    ledger = reconcile_attempts(jobs, recorder.events)
    record = {
        "arm": "A",
        "scene_id": "scene-1",
        "road_id": "road-scene-1",
        "stratum": "urban",
        "replicate_id": 0,
        "value": 0.8,
        "completed": True,
        "deviation_m": 0.5,
        "seed": None,
    }
    report = describe_validity(ledger, [observation(jobs[0], "att-1")], [record])
    assert "records_on_rejected_batch" in codes(report)
    assert report["status"] == "blocked"
    assert report["records"]["bound"] == 0
    assert no_admission_anywhere(report)


def test_record_deviation_must_match_full_stream_observation():
    ledger, observations = single_arm([4.2])
    record = {
        "arm": "A",
        "scene_id": "scene-0",
        "road_id": "road-scene-0",
        "stratum": "urban",
        "replicate_id": 0,
        "value": 0.8,
        "completed": True,
        "deviation_m": 0.2,
        "seed": None,
    }
    report = describe_validity(ledger, observations, [record])
    assert "record_observation_mismatch" in codes(report)
    assert report["deviation_bins"]["counts"]["[4,inf)"] == 1


def test_observation_absent_from_records_gets_reason():
    ledger, observations = single_arm([0.5, 0.5])
    record = {
        "arm": "A",
        "scene_id": "scene-0",
        "road_id": "road-scene-0",
        "stratum": "urban",
        "replicate_id": 0,
        "value": 0.8,
        "completed": True,
        "deviation_m": 0.5,
        "seed": None,
    }
    report = describe_validity(ledger, observations, [record])
    excluded = {row["job_id"]: row["reasons"] for row in report["excluded_units"]}
    assert excluded == {"synthetic-A-scene-1-0-aa": ["not_in_normalized_records"]}


def test_record_join_is_ambiguous_across_pilot_modes():
    jobs = [
        make_job("scene-1", protocol=PROTOCOL, phase="harmonizer_pilot"),
        make_job("scene-1", protocol=OTHER_PROTOCOL, phase="harmonizer_pilot"),
    ]
    recorder = Recorder()
    recorder.attempt(jobs[0], "att-1")
    recorder.attempt(jobs[1], "att-2")
    ledger = reconcile_attempts(jobs, recorder.events)
    record = {
        "arm": "A",
        "scene_id": "scene-1",
        "road_id": "road-scene-1",
        "stratum": "urban",
        "replicate_id": 0,
        "value": 0.8,
        "completed": True,
        "deviation_m": 0.5,
        "seed": None,
    }
    observations = [observation(jobs[0], "att-1"), observation(jobs[1], "att-2")]
    report = describe_validity(ledger, observations, [record], scope="synthetic")
    assert "record_binding_ambiguous" in codes(report)


def pilot_record(scene_id, deviation=0.5):
    return {
        "arm": "A",
        "scene_id": scene_id,
        "road_id": f"road-{scene_id}",
        "stratum": "urban",
        "replicate_id": 0,
        "value": 0.8,
        "completed": True,
        "deviation_m": deviation,
        "seed": None,
    }


def pilot_two_protocols(deviations=(0.5, 4.5)):
    jobs = [
        make_job("scene-1", protocol=protocol, phase="harmonizer_pilot")
        for protocol in (PROTOCOL, OTHER_PROTOCOL)
    ]
    recorder = Recorder()
    observations = []
    for index, (job, deviation) in enumerate(zip(jobs, deviations)):
        recorder.attempt(job, f"att-{index}")
        observations.append(observation(job, f"att-{index}", deviation))
    return reconcile_attempts(jobs, recorder.events), observations


def batch_entry(protocol, records):
    """The ``extract_bundle`` batch shape: arm, protocol_hash, policy_id, normalization."""
    return {
        "arm": "A",
        "protocol_hash": protocol,
        "policy_id": "teacher",
        "normalization": {"records": records, "admitted": True, "issues": []},
    }


def test_batch_entries_disambiguate_pilot_protocols():
    ledger, observations = pilot_two_protocols()
    batches = [
        batch_entry(PROTOCOL, [pilot_record("scene-1", 0.5)]),
        batch_entry(OTHER_PROTOCOL, [pilot_record("scene-1", 4.5)]),
    ]
    report = describe_validity(ledger, observations, batches, scope="synthetic")
    assert report["records"]["bound"] == 2
    assert report["records"]["out_of_envelope"] == 1
    assert not {"record_binding_ambiguous", "record_observation_mismatch"} & codes(report)
    assert report["excluded_units"] == []
    assert report["status"] == "passed"


def test_batch_entry_with_swapped_protocol_is_detected():
    ledger, observations = pilot_two_protocols()
    batches = [
        batch_entry(PROTOCOL, [pilot_record("scene-1", 4.5)]),
        batch_entry(OTHER_PROTOCOL, [pilot_record("scene-1", 0.5)]),
    ]
    report = describe_validity(ledger, observations, batches, scope="synthetic")
    assert "record_observation_mismatch" in codes(report)


@pytest.mark.parametrize(
    "entry,code",
    [
        (batch_entry("c" * 64, [pilot_record("scene-1")]), "record_unbound"),
        ({**batch_entry(PROTOCOL, []), "normalization": {"records": "x"}}, "invalid_record"),
        ({**batch_entry(PROTOCOL, [{**pilot_record("scene-1"), "arm": "B"}])}, "invalid_record"),
        (
            batch_entry(PROTOCOL, [{**pilot_record("scene-1"), "replicate_id": "0"}]),
            "invalid_record",
        ),
    ],
)
def test_batch_entry_negative_controls(entry, code):
    ledger, observations = pilot_two_protocols()
    report = describe_validity(ledger, observations, [entry], scope="synthetic")
    assert code in codes(report)
    assert report["status"] == "blocked"


def test_end_to_end_batches_from_extract_bind(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    extracted = extract_bundle(tmp_path, request)
    report = describe_validity(extracted["ledger"], extracted["observations"], extracted["batches"])
    assert report["records"]["bound"] == 1
    assert report["status"] == "passed"


def test_ledger_count_tampering_fails_closed():
    ledger, observations = single_arm([0.5])
    ledger = deepcopy(ledger)
    ledger["counts"]["completed_attempts"] = 0
    report = describe_validity(ledger, observations)
    assert "ledger_counts_mismatch" in codes(report)
    assert report["status"] == "blocked"


def test_selected_attempt_without_admission_is_inconsistent():
    ledger, observations = single_arm([0.5])
    ledger = deepcopy(ledger)
    ledger["admitted"] = False
    report = describe_validity(ledger, observations)
    assert "ledger_inconsistent" in codes(report)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda o: o.update(attempt_id="ghost"), "observation_unbound"),
        (lambda o: o.update(job_id="other-job"), "observation_job_mismatch"),
        (lambda o: o.update(run_uuid="other-run"), "observation_identity_mismatch"),
        (lambda o: o.update(requested_steps=7), "observation_requested_steps_mismatch"),
        (lambda o: o.update(surprise=1), "unknown_observation_field"),
        (lambda o: o.pop("attempt_id"), "invalid_observation"),
    ],
)
def test_observation_binding_negative_controls(mutate, expected):
    ledger, observations = single_arm([0.5])
    mutate(observations[0])
    report = describe_validity(ledger, observations)
    assert expected in codes(report)
    assert report["status"] == "blocked"
    assert report["units"][0]["deviation_bin"] == "unknown"
    reasons = report["excluded_units"][0]["reasons"]
    assert "observation_missing" in reasons or "observation_conflict" in reasons


def test_duplicate_observation_for_one_attempt_is_a_conflict():
    ledger, observations = single_arm([0.5])
    observations.append(deepcopy(observations[0]))
    observations[1]["full_max_deviation_m"] = 4.5
    report = describe_validity(ledger, observations)
    assert "duplicate_observation" in codes(report)
    assert report["units"][0]["deviation_bin"] == "unknown"
    assert "observation_conflict" in report["excluded_units"][0]["reasons"]


# --- artifact contract, scope and determinism --------------------------------------


def test_artifact_contract_fields_and_inputs_unchanged():
    ledger, observations = single_arm([0.5, 4.5])
    original = deepcopy((ledger, observations))
    report = describe_validity(ledger, observations)
    assert (ledger, observations) == original
    assert report["artifact_kind"] == "s2_validity_report"
    assert report["schema_version"] == 1
    assert report["producer"]["module"] == "gate.s2_diagnostics"
    assert len(report["producer"]["source_sha256"]) == 64
    assert set(report["input_hashes"]) == {
        "ledger_report",
        "observations",
        "records",
        "validity_definition",
    }
    assert report["input_hashes"]["records"] is None
    assert report["units_of_measure"]["deviation"] == "m"
    assert set(INDICATORS) == set(report["indicators"])
    for item in report["issues"]:
        assert set(item) == {"code", "path", "severity", "evidence_refs", "message"}
    json.dumps(report, allow_nan=False)


def test_scope_must_be_declared_for_non_synthetic_phases():
    jobs = [make_job("scene-1", phase="harmonizer_pilot")]
    recorder = Recorder()
    recorder.attempt(jobs[0], "att-1")
    ledger = reconcile_attempts(jobs, recorder.events)
    with pytest.raises(ValueError, match="scope"):
        describe_validity(ledger, [observation(jobs[0], "att-1")])
    report = describe_validity(ledger, [observation(jobs[0], "att-1")], scope="actual")
    assert report["scope"] == "actual"
    with pytest.raises(ValueError, match="scope"):
        describe_validity(ledger, [], scope="real")


def test_actual_scope_over_synthetic_phase_is_flagged():
    ledger, observations = single_arm([0.5])
    report = describe_validity(ledger, observations, scope="actual")
    assert "scope_phase_mismatch" in codes(report)


@pytest.mark.parametrize(
    "ledger_mutation",
    [
        lambda r: r.pop("counts"),
        lambda r: r.update(extra=1),
        lambda r: r.update(jobs="nope"),
        lambda r: r["jobs"][0].pop("status"),
        lambda r: r["attempts"][0].pop("state"),
        lambda r: r["jobs"][0].update(phase=["synthetic"]),
        lambda r: r["jobs"][0].update(replicate_id=True),
        lambda r: r["attempts"][0].update(upstream_identity="run"),
    ],
)
def test_malformed_ledger_report_fails_closed(ledger_mutation):
    ledger, observations = single_arm([0.5])
    ledger = deepcopy(ledger)
    ledger_mutation(ledger)
    with pytest.raises(ValueError):
        describe_validity(ledger, observations)


def test_observations_must_be_a_list():
    ledger, _ = single_arm([0.5])
    with pytest.raises(ValueError):
        describe_validity(ledger, {"not": "a list"})


def test_observation_and_record_order_do_not_change_the_report():
    fixture = load_pairs_fixture()
    ledger, observations = paired(fixture["pairs"])
    baseline = describe_validity(
        ledger, observations, validity_definition=fixture["validity_definition"]
    )
    shuffled = observations[:]
    random.Random(7).shuffle(shuffled)
    again = describe_validity(ledger, shuffled, validity_definition=fixture["validity_definition"])
    assert identity_hash(again) == identity_hash(baseline)
    assert (
        describe_validity(ledger, observations, validity_definition=fixture["validity_definition"])
        == baseline
    )


def test_record_order_does_not_change_the_report():
    ledger, observations = single_arm([0.5, 1.5, 4.5])
    records = [
        {**pilot_record(f"scene-{i}", deviation), "road_id": f"road-scene-{i}"}
        for i, deviation in enumerate([0.5, 1.5, 4.5])
    ]
    baseline = describe_validity(ledger, observations, records)
    assert baseline["records"]["bound"] == 3
    again = describe_validity(ledger, observations, list(reversed(records)))
    assert identity_hash(again) == identity_hash(baseline)


# --- review regressions (A5-F3..F10, P1, P3) ---------------------------------------


def unit_by_position(report, position):
    return next(unit for unit in report["units"] if unit["position"] == position)


def issue_paths(report, code):
    return {item["path"] for item in report["issues"] if item["code"] == code}


def test_selection_of_another_jobs_attempt_is_inconsistent_and_undescribed():
    ledger, observations = single_arm([0.5, 4.5])
    ledger = deepcopy(ledger)
    ledger["jobs"][0]["selected_attempt_id"] = ledger["jobs"][1]["selected_attempt_id"]
    report = describe_validity(ledger, observations)
    job_x, job_y = ledger["jobs"][0]["job_id"], ledger["jobs"][1]["job_id"]
    assert f"job:{job_x}" in issue_paths(report, "ledger_inconsistent")
    assert report["status"] == "blocked"
    unit = unit_by_position(report, 0)
    assert unit["described_attempt_id"] is None
    assert unit["deviation_bin"] == "unknown"
    assert "selection_inconsistent" in unit["reasons"]
    # Y's 4.5 m observation is counted once, for Y only.
    assert report["out_of_envelope"]["job_ids"] == [job_y]
    assert report["denominators"]["selected_observations"] == 1


def test_forged_admission_selecting_a_retry_is_inconsistent():
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1", kind="failed")
    recorder.attempt(job, "att-2", parent="att-1")
    ledger = deepcopy(reconcile_attempts([job], recorder.events))
    ledger.update(admitted=True, issues=[])
    ledger["jobs"][0]["selected_attempt_id"] = "att-2"
    report = describe_validity(ledger, [observation(job, "att-2", 0.5)])
    assert f"job:{job['job_id']}" in issue_paths(report, "ledger_inconsistent")
    assert report["units"][0]["described_attempt_id"] is None
    assert report["denominators"]["selected_observations"] == 0
    assert "selection_inconsistent" in report["excluded_units"][0]["reasons"]
    assert report["status"] == "blocked"


def test_foreign_selection_is_named_even_without_admission():
    ledger, observations = single_arm([0.5, 0.5])
    ledger = deepcopy(ledger)
    ledger.update(admitted=False, issues=[{"code": "x", "path": "events", "message": "m"}])
    ledger["jobs"][0]["selected_attempt_id"] = ledger["jobs"][1]["attempt_ids"][0]
    ledger["jobs"][1]["selected_attempt_id"] = None
    report = describe_validity(ledger, observations)
    messages = {
        item["message"] for item in report["issues"] if item["code"] == "ledger_inconsistent"
    }
    assert messages == {"Selected attempt is not one of the job's attempts."}


def test_forged_admission_of_a_multi_attempt_job_selecting_its_first_attempt():
    # Selection equals attempt_ids[0], both attempts completed and the status forged to
    # completed: only the sole-attempt requirement of W2 v0 admission can catch it.
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1")
    recorder.attempt(job, "att-2", parent="att-1")
    ledger = deepcopy(reconcile_attempts([job], recorder.events))
    ledger.update(admitted=True, issues=[])
    ledger["jobs"][0].update(selected_attempt_id="att-1", status="completed")
    observations = [observation(job, "att-1", 0.5), observation(job, "att-2", 0.5)]
    report = describe_validity(ledger, observations)
    assert f"job:{job['job_id']}" in issue_paths(report, "ledger_inconsistent")
    assert report["units"][0]["described_attempt_id"] is None


def forged_failed_sole_attempt():
    """A W2 ledger whose sole attempt failed, forged to admitted with no issues.

    Only the selection and the job status are forged; the attempt stays ``failed`` and
    the W2 counts stay true, so the job status check alone cannot catch the forgery.
    """
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1", kind="failed")
    ledger = deepcopy(reconcile_attempts([job], recorder.events))
    assert ledger["jobs"][0]["attempt_ids"] == ["att-1"]
    assert ledger["jobs"][0]["status"] == "failed"
    ledger.update(admitted=True, issues=[])
    ledger["jobs"][0].update(selected_attempt_id="att-1", status="completed")
    return ledger, [observation(job, "att-1", 0.5)]


def forged_failed_job_status():
    """A genuinely admitted sole completed attempt whose job status is forged to failed.

    The attempt state stays ``completed``, so the attempt state check alone cannot catch
    the forgery; only the job status clause can.
    """
    ledger, observations = single_arm([0.5])
    ledger = deepcopy(ledger)
    assert ledger["admitted"] is True and ledger["issues"] == []
    ledger["jobs"][0]["status"] = "failed"
    return ledger, observations


@pytest.mark.parametrize("forge", [forged_failed_sole_attempt, forged_failed_job_status])
def test_forged_admission_requires_completed_status_and_attempt_state(forge):
    # The two clauses of the W2 v0 'sole completed attempt' check are independent: each
    # input makes exactly one of them false, so removing either clause lets one case pass.
    ledger, observations = forge()
    job_id = ledger["jobs"][0]["job_id"]
    report = describe_validity(ledger, observations)
    assert f"job:{job_id}" in issue_paths(report, "ledger_inconsistent")
    unit = report["units"][0]
    assert unit["described_attempt_id"] is None
    assert unit["deviation_bin"] == "unknown"
    assert "selection_inconsistent" in unit["reasons"]
    assert report["denominators"]["selected_observations"] == 0
    assert report["status"] == "blocked"
    assert "ledger_counts_mismatch" not in codes(report)


def test_selection_without_admission_is_flagged_per_job():
    # admitted == (not issues) holds, so only the per-job selection check can fire.
    ledger, observations = single_arm([0.5, 0.5])
    ledger = deepcopy(ledger)
    ledger["admitted"] = False
    ledger["issues"] = [{"code": "synthetic_issue", "path": "events", "message": "m"}]
    ledger["jobs"][1]["selected_attempt_id"] = None
    report = describe_validity(ledger, observations)
    job_id = ledger["jobs"][0]["job_id"]
    assert issue_paths(report, "ledger_inconsistent") == {f"job:{job_id}"}
    assert "selection_inconsistent" in unit_by_position(report, 0)["reasons"]
    assert "selection_inconsistent" not in unit_by_position(report, 1)["reasons"]


def test_retry_out_of_envelope_attempt_is_listed_separately():
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1", kind="failed")
    recorder.attempt(job, "att-2", parent="att-1")
    ledger = reconcile_attempts([job], recorder.events)
    report = describe_validity(ledger, [observation(job, "att-2", 4.5)])
    section = report["out_of_envelope"]
    assert section["count"] == 0
    assert section["job_ids"] == []
    assert section["attempt_count"] == 1
    assert section["attempt_ids"] == ["att-2"]
    assert section["undescribed_attempt_ids"] == ["att-2"]
    assert "observation:att-2" in issue_paths(report, "out_of_envelope_attempt_not_described")


def test_batches_from_a_rejected_extraction_are_flagged_not_crashing(tmp_path):
    request, _, trace, _ = fixture_bundle(tmp_path)
    trace = deepcopy(trace)
    trace["samples"][2]["black_frame"] = True
    request["artifacts"][0]["trace"] = json_receipt(tmp_path, "trace.json", trace)
    extracted = extract_bundle(tmp_path, request)
    assert extracted["admitted"] is False
    assert extracted["ledger"]["admitted"] is True
    report = describe_validity(extracted["ledger"], extracted["observations"], extracted["batches"])
    assert "records_on_rejected_batch" in codes(report)
    assert report["status"] == "blocked"
    assert report["records"]["bound"] == 0
    assert report["upstream_acceptance"] == {
        "w2_ledger": "admitted",
        "w3_normalization": "rejected",
    }
    assert "normalized_records_unusable" in report["excluded_units"][0]["reasons"]
    assert report["input_hashes"]["records"] is not None
    assert no_admission_anywhere(report)


def test_non_admitted_normalization_records_are_not_used():
    ledger, observations = single_arm([0.5])
    entry = batch_entry(PROTOCOL, [{**pilot_record("scene-0"), "road_id": "road-scene-0"}])
    entry["normalization"]["admitted"] = False
    report = describe_validity(ledger, observations, [entry])
    assert "records_on_rejected_batch" in codes(report)
    assert report["records"]["bound"] == 0
    assert report["status"] == "blocked"
    assert report["upstream_acceptance"]["w3_normalization"] == "rejected"


def rejected_entry(issues):
    entry = batch_entry(PROTOCOL, [{**pilot_record("scene-0"), "road_id": "road-scene-0"}])
    entry["normalization"].update(admitted=False, issues=issues)
    return entry


def test_records_hash_covers_normalization_issues():
    # The issues decide invalid_record, so two inputs differing only there must hash apart.
    ledger, observations = single_arm([0.5])
    adapter_issue = {"code": "x", "identity": ("run", "clip", "roll"), "message": "m"}
    empty = describe_validity(ledger, observations, [rejected_entry([])])
    flagged = describe_validity(ledger, observations, [rejected_entry([adapter_issue])])
    assert empty["input_hashes"]["records"] != flagged["input_hashes"]["records"]
    json.dumps(flagged, allow_nan=False)
    # A tuple identity hashes as its JSON list: the projection is (code, message, identity).
    as_list = {**adapter_issue, "identity": ["run", "clip", "roll"]}
    listed = describe_validity(ledger, observations, [rejected_entry([as_list])])
    assert listed["input_hashes"]["records"] == flagged["input_hashes"]["records"]
    other = {**adapter_issue, "message": "n"}
    changed = describe_validity(ledger, observations, [rejected_entry([other])])
    assert changed["input_hashes"]["records"] != flagged["input_hashes"]["records"]
    # The code and the identity are each in the projection.
    for variant in (
        {**adapter_issue, "code": "y"},
        {**adapter_issue, "identity": ("run", "clip", "other")},
    ):
        varied = describe_validity(ledger, observations, [rejected_entry([variant])])
        assert varied["input_hashes"]["records"] != flagged["input_hashes"]["records"]
    # Adapter-irrelevant normalization keys are still not hashed.
    entry = rejected_entry([adapter_issue])
    entry["normalization"]["diagnostics"] = {"note": "ignored"}
    ignored = describe_validity(ledger, observations, [entry])
    assert ignored["input_hashes"]["records"] == flagged["input_hashes"]["records"]


def test_records_hash_distinguishes_absent_and_empty_issues():
    ledger, observations = single_arm([0.5])
    absent = rejected_entry([])
    del absent["normalization"]["issues"]
    a = describe_validity(ledger, observations, [absent])["input_hashes"]["records"]
    b = describe_validity(ledger, observations, [rejected_entry([])])["input_hashes"]["records"]
    assert a != b


def test_records_count_is_every_supplied_record_whatever_its_use():
    # Bare record over a W2-rejected batch: supplied 1, used 0.
    jobs = [make_job("scene-1"), make_job("scene-2")]
    recorder = Recorder()
    recorder.attempt(jobs[0], "att-1")
    rejected_ledger = reconcile_attempts(jobs, recorder.events)
    bare = {**pilot_record("scene-1")}
    report = describe_validity(rejected_ledger, [observation(jobs[0], "att-1")], [bare])
    assert report["records"] == {"supplied": True, "count": 1, "bound": 0, "out_of_envelope": 0}
    # Non-admitted batch entry over an admitted W2 ledger: supplied 1, used 0 (was 0, 0).
    ledger, observations = single_arm([0.5])
    report = describe_validity(ledger, observations, [rejected_entry([])])
    assert report["records"] == {"supplied": True, "count": 1, "bound": 0, "out_of_envelope": 0}
    # Admitted batch entry with two records, one unbound: supplied 2, bound 1.
    entry = batch_entry(
        PROTOCOL,
        [
            {**pilot_record("scene-0"), "road_id": "road-scene-0"},
            {**pilot_record("scene-9"), "road_id": "road-scene-9"},
        ],
    )
    report = describe_validity(ledger, observations, [entry])
    assert report["records"] == {"supplied": True, "count": 2, "bound": 1, "out_of_envelope": 0}
    assert "record_unbound" in codes(report)
    # A malformed entry that still carries a records list counts that list.
    odd = batch_entry(PROTOCOL, [{**pilot_record("scene-0"), "road_id": "road-scene-0"}])
    odd["normalization"].update(admitted="yes", issues=[])
    report = describe_validity(ledger, observations, [odd])
    assert report["records"]["count"] == 1
    assert report["status"] == "blocked"
    # An entry whose records value is not a list cannot be counted: missing is not zero,
    # so the whole count is null even beside a countable entry.
    malformed = {**batch_entry(PROTOCOL, []), "normalization": {"records": "x"}}
    report = describe_validity(ledger, observations, [malformed])
    assert report["records"]["count"] is None
    no_dict = {**batch_entry(PROTOCOL, []), "normalization": "x"}
    report = describe_validity(ledger, observations, [entry, no_dict])
    assert report["records"]["count"] is None
    assert report["status"] == "blocked"
    assert report["upstream_acceptance"]["w3_normalization"] == "invalid"
    json.dumps(report, allow_nan=False)
    assert describe_validity(ledger, observations)["records"]["count"] == 0
    assert describe_validity(ledger, observations)["records"]["supplied"] is False
    assert "supplied" in report["units_of_measure"]["records"]


@pytest.mark.parametrize(
    "normalization",
    [
        {"records": []},
        {"records": [], "admitted": "yes"},
        {"records": [], "admitted": True, "issues": [{"code": "x"}]},
    ],
)
def test_normalization_requires_boolean_admission_consistent_with_issues(normalization):
    ledger, observations = single_arm([0.5])
    entry = {**batch_entry(PROTOCOL, []), "normalization": normalization}
    report = describe_validity(ledger, observations, [entry])
    assert "invalid_record" in codes(report)
    assert report["status"] == "blocked"


def test_upstream_acceptance_is_explicit():
    ledger, observations = single_arm([0.5])
    report = describe_validity(ledger, observations)
    assert report["upstream_acceptance"] == {
        "w2_ledger": "admitted",
        "w3_normalization": "not_supplied",
    }
    bare = {**pilot_record("scene-0"), "road_id": "road-scene-0"}
    assert describe_validity(ledger, observations, [bare])["upstream_acceptance"] == {
        "w2_ledger": "admitted",
        "w3_normalization": "not_stated",
    }
    entry = batch_entry(PROTOCOL, [bare])
    assert describe_validity(ledger, observations, [entry])["upstream_acceptance"] == {
        "w2_ledger": "admitted",
        "w3_normalization": "admitted",
    }


def test_huge_integer_deviation_is_invalid_not_overflow():
    ledger, observations = single_arm([0.5])
    observations[0]["full_max_deviation_m"] = 10**400
    report = describe_validity(ledger, observations)
    assert "invalid_deviation" in codes(report)
    assert report["deviation_bins"]["counts"]["unknown"] == 1
    assert deviation_bin(10**400) == "unknown"


@pytest.mark.parametrize("field,value", [("code", 7), ("path", None), ("path", 3)])
def test_non_text_ledger_issue_fails_closed(field, value):
    ledger, observations = single_arm([0.5])
    ledger = deepcopy(ledger)
    ledger["admitted"] = False
    ledger["jobs"][0]["selected_attempt_id"] = None
    item = {"code": "x", "path": f"job:{ledger['jobs'][0]['job_id']}", "message": "m"}
    item[field] = value
    ledger["issues"] = [item]
    with pytest.raises(ValueError):
        describe_validity(ledger, observations)


def test_outside_table_names_its_units():
    jobs = [
        make_job("s1", arm="A"),
        {**make_job("s1", arm="B"), "job_id": "b1"},
        {**make_job("s1", arm="B"), "job_id": "b2"},
    ]
    recorder, observations = Recorder(), []
    for index, job in enumerate(jobs):
        recorder.attempt(job, f"a{index}")
        observations.append(observation(job, f"a{index}", 0.5))
    report = describe_validity(
        reconcile_attempts(jobs, recorder.events), observations, validity_definition=ENVELOPE_ONLY
    )
    assert report["validity_table"]["outside_table"] == {"unknown_pairs": 0, "unpaired_jobs": 3}
    assert "unpaired_jobs" in report["units_of_measure"]["validity_table"]


def test_duplicate_job_id_reasons_stay_with_their_row():
    job = make_job("scene-1")
    recorder = Recorder()
    recorder.attempt(job, "att-1")
    ledger = reconcile_attempts([job, dict(job)], recorder.events)
    report = describe_validity(ledger, [observation(job, "att-1", 0.5)])
    assert unit_by_position(report, 0)["reasons"] == ["batch_not_admitted", "duplicate_job_id"]
    assert unit_by_position(report, 1)["reasons"] == [
        "batch_not_admitted",
        "duplicate_job_id",
        "missing_job",
    ]


def test_rejected_second_observation_marks_described_unit_conflict():
    ledger, observations = single_arm([0.5])
    observations.append({**deepcopy(observations[0]), "surprise": 1})
    report = describe_validity(ledger, observations)
    unit = report["units"][0]
    assert unit["described_attempt_id"] == "att-0"
    assert unit["deviation_bin"] == "[0,1)"
    assert "observation_conflict" in unit["reasons"]
    assert "unknown_observation_field" in codes(report)


def test_pairing_convention_is_labelled_as_agent_default():
    ledger, observations = paired(load_pairs_fixture()["pairs"])
    pairing = describe_validity(ledger, observations)["validity_table"]["pairing"]
    assert pairing["key"] == ["phase", "scene_id", "replicate_id", "protocol_hash"]
    assert pairing["status"] == "agent_technical_default"
    assert pairing["owner_convention"] is None
