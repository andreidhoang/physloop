"""Known answers and negative controls for gate.s2_telemetry (A6; readiness CPU-07).

Hand-derived oracles: two GPUs for 1,800 s = 1 GPU-hour; four useful episodes
then give 8 per wall-hour and 4 per GPU-hour; constant 100 W over 10 s = 1,000 J
by the trapezoid rule; a 0 -> 100 W ramp over 10 s = 500 J; a counter 1000,
3000, 500, 1500 mJ has one reset and a lower-bound increase of
2000 + 500 + 1000 = 3500 mJ.
"""

import copy
import random

import pytest

from gate import s2_telemetry as tel

S = 1_000_000  # microseconds per second
CLOCK = {"clock_id": "host-mono-1", "clock_kind": "monotonic", "clock_scope": "host",
         "utc_anchor": "2026-11-08T00:00:00Z"}


def allocation(end_s=1800, segments=None, gpus=("gpu0", "gpu1")):
    if segments is None:
        segments = [("startup", 0, 300), ("warm_up", 300, 600), ("active", 600, 1500),
                    ("idle", 1500, 1700), ("tail", 1700, 1800)]
    return {
        "allocation_id": "alloc-1",
        "clock_id": "host-mono-1",
        "gpu_count": len(gpus),
        "devices": list(gpus),
        "billed_start_us": 0,
        "billed_end_us": end_s * S,
        "segments": [{"phase": p, "start_us": a * S, "end_us": b * S} for p, a, b in segments],
    }


def rollout(i, status="completed", useful=True, start=700, end=800):
    return {
        "attempt_id": f"att-{i}",
        "job_id": f"job-{i}",
        "clock_id": "host-mono-1",
        "start_us": start * S,
        "end_us": None if end is None else end * S,
        "status": status,
        "useful": useful,
    }


def sample(metric, t_s, value, status="ok", device="gpu0", t_us=None):
    return {
        "metric": metric,
        "device": device,
        "clock_id": "host-mono-1",
        "t_us": t_us if t_us is not None else int(t_s * S),
        "status": status,
        "value": value if status == "ok" else None,
        "raw_ref": f"prom/sample-{t_s}.prom",
    }


def run(samples=(), alloc=None, rollouts=(), gap_s=2, **kwargs):
    return tel.summarize_telemetry(
        list(samples),
        alloc or allocation(),
        list(rollouts),
        clock_metadata=kwargs.pop("clock_metadata", CLOCK),
        max_sample_gap_us=gap_s * S,
        scope="synthetic",
        **kwargs,
    )


def codes(report):
    return {item["code"] for item in report["issues"]}


POWER = "DCGM_FI_DEV_POWER_USAGE"
ENERGY = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
FB = "DCGM_FI_DEV_FB_USED"
XID = "DCGM_FI_DEV_XID_ERRORS"


# --- allocation partition and useful work ----------------------------------------


def test_gpu_hours_and_useful_episode_rates_known_answer():
    rollouts = [rollout(i) for i in range(4)]
    rollouts += [rollout(8, status="failed", useful=False),
                 rollout(9, status="completed", useful=False)]
    report = run(rollouts=rollouts)
    assert report["status"] == "passed", report["issues"]
    alloc = report["allocation"]
    assert alloc["billed_us"] == 1_800_000_000
    assert alloc["billed_hours"]["exact"] == "1/2"
    assert alloc["gpu_hours"]["exact"] == "1" and alloc["gpu_hours"]["value"] == 1.0
    work = report["useful_work"]
    assert work["useful_episodes"] == 4 and work["useful_unknown"] == 0
    assert work["per_wall_hour"]["exact"] == "8"
    assert work["per_gpu_hour"]["exact"] == "4"
    assert work["status"] == "passed"
    assert work["attempts_by_status"] == {"completed": 5, "failed": 1, "cancelled": 0,
                                          "unresolved": 0}


def test_partition_sums_exactly_to_billed_time():
    partition = run()["allocation"]["partition"]
    assert partition["status"] == "passed"
    assert partition["unaccounted_us"] == 0
    phases = partition["phases"]
    assert sum(item["us"] for item in phases.values()) == 1_800_000_000
    assert phases["startup"]["share"]["exact"] == "1/6"
    assert phases["active"]["us"] == 900 * S
    assert phases["active"]["gpu_hours"]["exact"] == "1/2"
    assert phases["retry"]["us"] == 0 and phases["acquisition"]["us"] == 0


@pytest.mark.parametrize(
    "segments, code",
    [
        ([("startup", 0, 300), ("active", 400, 1800)], "partition_gap"),
        ([("startup", 0, 300), ("active", 200, 1800)], "partition_overlap"),
        ([("startup", 0, 300), ("active", 300, 1900)], "partition_out_of_bounds"),
        ([("startup", 0, 300), ("active", 300, 1700)], "partition_gap"),
        ([("startup", 0, 300), ("bogus", 300, 1800)], "allocation_invalid"),
        ([("startup", 0, 0), ("active", 0, 1800)], "allocation_invalid"),
    ],
)
def test_partition_negative_controls(segments, code):
    report = run(alloc=allocation(segments=segments))
    assert report["status"] == "blocked"
    assert code in codes(report)
    assert report["allocation"]["partition"]["status"] == "blocked"


def test_partition_gap_is_reported_as_unaccounted_not_padded():
    report = run(alloc=allocation(segments=[("startup", 0, 300), ("active", 400, 1800)]))
    partition = report["allocation"]["partition"]
    assert partition["unaccounted_us"] == 100 * S
    assert partition["unaccounted_intervals_us"] == [[300 * S, 400 * S]]


def test_useful_unknown_is_not_counted_and_marks_rates_unverified():
    report = run(rollouts=[rollout(0), rollout(1, useful=None)])
    work = report["useful_work"]
    assert work["useful_episodes"] == 1 and work["useful_unknown"] == 1
    assert work["status"] == "unverified"
    assert work["per_gpu_hour"]["exact"] == "1"
    assert work["rates_are_lower_bounds"] is True


@pytest.mark.parametrize(
    "bad, code",
    [
        (rollout(0, status="failed", useful=True), "rollout_invalid"),
        (rollout(0, start=1799, end=1801), "rollout_outside_allocation"),
        ({**rollout(0), "clock_id": "other"}, "clock_scope_mismatch"),
        ({**rollout(0), "exit_code": 0}, "rollout_invalid"),
        (rollout(0, end=None), "rollout_invalid"),
        (rollout(0, start=800, end=700), "rollout_invalid"),
    ],
)
def test_rollout_negative_controls(bad, code):
    report = run(rollouts=[bad])
    assert code in codes(report)
    assert report["useful_work"]["useful_episodes"] == 0


def test_unresolved_rollout_without_end_is_allowed():
    report = run(rollouts=[rollout(0, status="unresolved", useful=None, end=None)])
    assert report["status"] == "passed"
    assert report["useful_work"]["attempts_by_status"]["unresolved"] == 1


def test_overlapping_service_time_is_never_billed_time():
    report = run(rollouts=[rollout(0, start=600, end=1200), rollout(1, start=900, end=1500)])
    service = report["service_time"]
    assert service["rollout_service_us_sum"] == 1200 * S
    assert service["rollout_service_union_us"] == 900 * S
    assert report["allocation"]["billed_us"] == 1800 * S


# --- energy --------------------------------------------------------------------


def ten_seconds():
    return allocation(end_s=10, segments=[("active", 0, 10)], gpus=("gpu0",))


def test_constant_100w_for_10s_is_1000j():
    samples = [sample(POWER, t, 100.0) for t in range(11)]
    energy = run(samples, alloc=ten_seconds())["energy"]["power_integration"][POWER]["gpu0"]
    assert energy["status"] == "passed"
    assert energy["total_energy_j"]["exact"] == "1000"
    assert energy["uncovered_intervals_us"] == []


def test_ramp_trapezoid_is_500j():
    samples = [sample(POWER, 0, 0.0), sample(POWER, 10, 100.0)]
    energy = run(samples, alloc=ten_seconds(), gap_s=10)["energy"]["power_integration"]
    assert energy[POWER]["gpu0"]["total_energy_j"]["exact"] == "500"


def test_gap_beyond_limit_gives_incomplete_energy_not_interpolation():
    samples = [sample(POWER, t, 100.0) for t in (0, 1, 2, 6, 7, 8, 9, 10)]
    report = run(samples, alloc=ten_seconds())
    energy = report["energy"]["power_integration"][POWER]["gpu0"]
    assert energy["status"] == "not_estimable"
    assert energy["total_energy_j"]["value"] is None
    assert energy["covered_energy_j"]["exact"] == "600"
    assert energy["uncovered_intervals_us"] == [[2 * S, 6 * S]]
    assert report["metrics"][POWER]["gpu0"]["availability"] == "incomplete"


def test_edges_without_samples_are_uncovered():
    samples = [sample(POWER, t, 100.0) for t in range(1, 10)]
    energy = run(samples, alloc=ten_seconds())["energy"]["power_integration"][POWER]["gpu0"]
    assert energy["status"] == "not_estimable"
    assert energy["covered_energy_j"]["exact"] == "800"
    assert energy["uncovered_intervals_us"] == [[0, S], [9 * S, 10 * S]]


def test_counter_energy_and_reset_is_not_negative():
    clean = [sample(ENERGY, 0, 1000.0), sample(ENERGY, 10, 11000.0)]
    counter = run(clean, alloc=ten_seconds(), gap_s=10)["energy"]["counter_delta"][ENERGY]["gpu0"]
    assert counter["status"] == "passed"
    assert counter["total_energy_j"]["exact"] == "10"
    assert counter["reset_events_us"] == []

    reset = [sample(ENERGY, t, v) for t, v in ((0, 1000.0), (5, 3000.0), (7, 500.0),
                                               (10, 1500.0))]
    report = run(reset, alloc=ten_seconds(), gap_s=10)
    counter = report["energy"]["counter_delta"][ENERGY]["gpu0"]
    assert counter["status"] == "unverified"
    assert counter["total_energy_j"]["value"] is None
    assert counter["lower_bound_energy_j"]["exact"] == "7/2"
    assert counter["reset_events_us"] == [7 * S]
    series = report["metrics"][ENERGY]["gpu0"]
    assert series["counter"]["resets"] == 1
    assert series["counter"]["increase_lower_bound"]["exact"] == "3500"


# --- availability, XID, memory -------------------------------------------------------


def test_no_data_is_not_zero_xids():
    samples = [sample(XID, t, None, status="no_data") for t in range(0, 11, 2)]
    report = run(samples, alloc=ten_seconds())
    xid = report["xid"]["gpu0"]
    assert xid["availability"] == "no_data"
    assert xid["error_count"] is None
    assert xid["observed_values"] == []
    assert xid["fault_observability"] == "unverified"
    assert report["metrics"][XID]["gpu0"]["status_counts"]["no_data"] == 6


def test_xid_absent_and_observed():
    report = run([], alloc=ten_seconds())
    assert report["xid"]["gpu0"]["availability"] == "absent"
    assert report["xid"]["gpu0"]["error_count"] is None
    observed = run([sample(XID, 3, 43)], alloc=ten_seconds())["xid"]["gpu0"]
    assert observed["observed_values"] == [43]
    assert observed["error_count"] is None


@pytest.mark.parametrize(
    "status, availability",
    [("not_supported", "unsupported"), ("scrape_failed", "failed"), ("no_permission", "failed")],
)
def test_availability_labels(status, availability):
    samples = [sample(POWER, t, None, status=status) for t in range(11)]
    report = run(samples, alloc=ten_seconds())
    assert report["metrics"][POWER]["gpu0"]["availability"] == availability
    energy = report["energy"]["power_integration"][POWER]["gpu0"]
    assert energy["status"] == "not_estimable" and energy["total_energy_j"]["value"] is None


def test_healthy_series_and_absent_metric_reported_per_device():
    samples = [sample(POWER, t, 50.0) for t in range(0, 11, 2)]
    report = run(samples, alloc=ten_seconds())
    assert report["metrics"][POWER]["gpu0"]["availability"] == "healthy"
    assert report["metrics"][FB]["gpu0"]["availability"] == "absent"
    assert report["metrics"][POWER]["gpu0"]["unit"] == "W"
    assert report["metrics"][ENERGY]["gpu0"]["unit"] == "mJ"


def test_memory_high_water_mark_is_sampled():
    samples = [sample(FB, 0, 100.0), sample(FB, 5, 5000.0), sample(FB, 10, 3000.0)]
    memory = run(samples, alloc=ten_seconds(), gap_s=5)["memory"][FB]["gpu0"]
    assert memory["high_water_mib"] == 5000.0
    assert memory["t_us"] == 5 * S
    assert "unobserved" in memory["limitation"]
    assert memory["max_observed_gap_us"] == 5 * S


# --- sample validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad, code",
    [
        ({**sample(POWER, 1, 10.0), "clock_id": "process-2"}, "clock_scope_mismatch"),
        (sample("DCGM_FI_DEV_UNKNOWN", 1, 1.0), "metric_not_pinned"),
        ({**sample(POWER, 1, 10.0), "value": None}, "sample_invalid"),
        ({**sample(POWER, 1, None, status="no_data"), "value": 0}, "sample_invalid"),
        ({**sample(POWER, 1, 10.0), "status": None}, "sample_invalid"),
        ({k: v for k, v in sample(POWER, 1, 10.0).items() if k != "status"}, "sample_invalid"),
        (sample(POWER, 1, float("inf")), "sample_invalid"),
        (sample(POWER, 1, True), "sample_invalid"),
        (sample(POWER, 1, 10.0, device="gpu7"), "device_not_allocated"),
        (sample(POWER, 11, 10.0), "sample_outside_allocation"),
    ],
)
def test_sample_negative_controls(bad, code):
    report = run([sample(POWER, 0, 10.0), bad], alloc=ten_seconds())
    assert code in codes(report)
    assert report["metrics"][POWER]["gpu0"]["status_counts"]["ok"] == 1


def test_duplicate_sample_rejected():
    report = run([sample(POWER, 1, 10.0), sample(POWER, 1, 11.0)], alloc=ten_seconds())
    assert "duplicate_sample" in codes(report)
    assert report["status"] == "blocked"


# --- registration, clocks and determinism -------------------------------------------


def test_stabilized_throughput_requires_registered_rule():
    report = run()
    assert report["stabilized_throughput"] == {
        "status": "blocked", "code": "stabilization_rule_unregistered", "value": None}
    assert report["status"] == "passed"
    report = run(stabilization_rule={"rule_id": "x"})
    assert report["stabilized_throughput"]["code"] == "unsupported_stabilization_rule"
    assert report["stabilized_throughput"]["value"] is None


def test_no_device_latency_claim():
    assert run()["device_latency"]["status"] == "not_applicable"


def test_clock_metadata_must_match_and_be_monotonic():
    report = run(clock_metadata={**CLOCK, "clock_id": "other"})
    assert "clock_scope_mismatch" in codes(report) and report["status"] == "blocked"
    with pytest.raises(ValueError):
        run(clock_metadata={**CLOCK, "clock_kind": "utc"})
    with pytest.raises(ValueError):
        tel.summarize_telemetry([], allocation(), [], clock_metadata=CLOCK,
                                max_sample_gap_us=0, scope="synthetic")
    with pytest.raises(ValueError):
        tel.summarize_telemetry([], allocation(), [], clock_metadata=CLOCK,
                                max_sample_gap_us=S, scope="real")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(gpu_count=3),
        lambda a: a.update(devices=["gpu0", "gpu0"]),
        lambda a: a.update(billed_end_us=0),
        lambda a: a.pop("segments"),
        lambda a: a.update(gpu_count=True),
    ],
)
def test_allocation_invalid(mutate):
    alloc = allocation()
    mutate(alloc)
    report = run(alloc=alloc)
    assert "allocation_invalid" in codes(report)
    assert report["status"] == "blocked"
    assert report["useful_work"]["per_gpu_hour"]["value"] is None


def test_reorder_invariant_and_utc_anchor_not_in_arithmetic():
    samples = [sample(POWER, t, 100.0 + t) for t in range(11)]
    samples += [sample(FB, t, 10.0 * t) for t in range(0, 11, 2)]
    rollouts = [rollout(i, start=1, end=2 + i) for i in range(3)]
    first = run(samples, alloc=ten_seconds(), rollouts=rollouts)
    shuffled = copy.deepcopy(samples)
    random.Random(3).shuffle(shuffled)
    second = run(shuffled, alloc=ten_seconds(), rollouts=list(reversed(rollouts)),
                 clock_metadata={**CLOCK, "utc_anchor": "2027-01-01T00:00:00Z"})
    for key in ("allocation", "useful_work", "energy", "metrics", "memory", "xid"):
        assert first[key] == second[key]
    assert first["input_hashes"]["samples"] == second["input_hashes"]["samples"]
    assert first["clock"]["utc_anchor"] != second["clock"]["utc_anchor"]
    assert first["artifact_kind"] == "s2_telemetry_summary"


# --- review regressions (A6-F2, F6, F7, F9) ---------------------------------------


def test_counter_gap_beyond_limit_is_unverified_not_passed():
    """A6-F2: a reset hidden in a gap over the limit is undetectable: never passed."""
    samples = [sample(ENERGY, 0, 1000.0), sample(ENERGY, 10, 11000.0)]
    report = run(samples, alloc=ten_seconds(), gap_s=2)
    counter = report["energy"]["counter_delta"][ENERGY]["gpu0"]
    assert report["metrics"][ENERGY]["gpu0"]["availability"] == "incomplete"
    assert counter["status"] == "unverified"
    assert counter["total_energy_j"] == {"status": "unverified", "value": None, "exact": None,
                                         "reason": "gap_beyond_coverage_limit"}
    assert counter["lower_bound_energy_j"]["exact"] == "10"
    assert counter["gaps_over_limit_us"] == [[0, 10 * S]]


@pytest.mark.parametrize("metric", [ENERGY, "DCGM_FI_DEV_PCIE_REPLAY_COUNTER"])
def test_series_counter_increase_across_gap_is_unverified(metric):
    """A6R-1: the per-series increase agrees with counter-delta energy across a long gap."""
    samples = [sample(metric, 0, 1000.0), sample(metric, 10, 11000.0)]
    counter = run(samples, alloc=ten_seconds(), gap_s=2)["metrics"][metric]["gpu0"]["counter"]
    assert counter["increase"] == {"status": "unverified", "value": None, "exact": None,
                                   "reason": "gap_beyond_coverage_limit"}
    assert counter["increase_lower_bound"]["exact"] == "10000"
    assert counter["resets"] == 0
    within = run(samples, alloc=ten_seconds(), gap_s=10)["metrics"][metric]["gpu0"]["counter"]
    assert within["increase"]["status"] == "passed"
    assert within["increase"]["exact"] == "10000"


def test_input_hashes_cover_every_supplied_row():
    """A6-F6: dropped samples and attempts change the input hash."""
    base = run([sample(POWER, 0, 1.0)], alloc=ten_seconds())
    more = run([sample(POWER, 0, 1.0), sample("UNPINNED", 0, 1.0)], alloc=ten_seconds())
    assert more["input_hashes"]["samples"] != base["input_hashes"]["samples"]
    assert more["input_hashes"]["accepted_samples"] == base["input_hashes"]["accepted_samples"]
    bad_attempt = {**rollout(1), "useful": "yes"}
    r1 = run(alloc=ten_seconds(), rollouts=[rollout(0, start=1, end=2)])
    r2 = run(alloc=ten_seconds(), rollouts=[rollout(0, start=1, end=2), bad_attempt])
    assert r2["input_hashes"]["rollouts"] != r1["input_hashes"]["rollouts"]
    assert r2["input_hashes"]["accepted_rollouts"] == r1["input_hashes"]["accepted_rollouts"]
    nonjson = run([sample(POWER, 0, 1.0), {"metric": float("nan")}], alloc=ten_seconds())
    assert nonjson["input_hashes"]["samples"] not in (None, base["input_hashes"]["samples"])
    assert nonjson["samples"]["non_json"] == 1


def test_all_supplied_samples_dropped_blocks_and_is_not_absent():
    """A6-F7: supplied-but-dropped telemetry is distinguishable from never supplied."""
    misnamed = [sample("dcgm_fi_dev_power_usage", t, 100.0) for t in range(11)]
    report = run(misnamed, alloc=ten_seconds())
    assert report["status"] == "blocked"
    assert "all_samples_dropped" in codes(report)
    assert report["samples"]["supplied"] == 11 and report["samples"]["accepted"] == 0
    assert report["samples"]["dropped_by_reason"] == {"metric_not_pinned": 11}

    outside = [sample(POWER, t, 100.0) for t in range(11, 22)]
    report = run(outside, alloc=ten_seconds())
    assert report["status"] == "blocked"
    series = report["metrics"][POWER]["gpu0"]
    assert series["availability"] == "dropped"
    assert series["dropped_samples"] == 11
    assert report["metrics"][FB]["gpu0"]["availability"] == "absent"
    assert report["metrics"][FB]["gpu0"]["dropped_samples"] == 0

    assert run([], alloc=ten_seconds())["status"] == "passed"


def test_rejected_attempts_make_rates_lower_bounds():
    """A6-F9: dropped attempts can only lower the useful count."""
    rollouts = [rollout(0, start=1, end=2), rollout(1, start=1, end=2),
                {**rollout(2, start=1, end=2), "attempt_id": "dup"},
                {**rollout(3, start=1, end=2), "attempt_id": "dup"}]
    report = run(alloc=ten_seconds(), rollouts=rollouts)
    work = report["useful_work"]
    assert report["status"] == "blocked"
    assert work["useful_episodes"] == 2
    assert work["rejected_attempts"] == 2
    assert work["rates_are_lower_bounds"] is True
    assert work["status"] == "blocked"
    clean = run(alloc=ten_seconds(), rollouts=rollouts[:2])["useful_work"]
    assert clean["rejected_attempts"] == 0 and clean["rates_are_lower_bounds"] is False
    assert clean["status"] == "passed"


def test_useful_flag_is_consumed_and_its_rule_labelled_an_agent_default():
    # F6 residual (c): the consumer never infers useful; the producer's rule carries the
    # D-018 agent-default label here as in gate.s2_telemetry_sources.
    assert "agent technical default (D-018)" in tel.__doc__
    assert "never\ninfers it" in tel.__doc__
