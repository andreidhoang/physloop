"""Fake-runner known answers for the fixed-interval nvidia-smi sampler (build-spec B1).

No test runs nvidia-smi: every query goes through an injected fake runner and clock.
"""

import json

import pytest

from gate.s2_telemetry import summarize_telemetry
from scripts.s2_gpu_sampler import (
    QUERY_ARGV,
    RAW_NAME,
    SUMMARY_NAME,
    load_samples,
    main,
    parse_query_output,
    sample,
    tick_samples,
)

GPU0 = "GPU-00000000-0000-0000-0000-000000000000"
GPU1 = "GPU-11111111-1111-1111-1111-111111111111"
# Header of ``--format=csv`` for the four queried fields. Every token is copied from a
# recorded host log (evidence/s2-vast-host-20261101-v1): "index" and "uuid" from
# probe-v2/gpu_identity.log, "power.draw [W]" from probe-v2/gpu_telemetry.log and
# "memory.used [MiB]" from monitor-v1/nvml-after.log, all ", "-separated.
HEADER = "index, uuid, power.draw [W], memory.used [MiB]"


class FakeClock:
    def __init__(self, start=10_000_000):
        self.now = start
        self.sleeps = []

    def monotonic_us(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += round(seconds * 1_000_000)


class FakeRunner:
    """Returns scripted results; each call may advance the clock (query duration)."""

    def __init__(self, clock, results, cost_us=100_000):
        self.clock, self.results, self.cost_us, self.calls = clock, list(results), cost_us, []

    def __call__(self, argv, timeout_s):
        self.calls.append((list(argv), timeout_s, self.clock.now))
        result = self.results.pop(0) if self.results else ok_result()
        self.clock.now += result.pop("cost_us", self.cost_us) if isinstance(result, dict) else 0
        if isinstance(result, Exception):
            raise result
        return result


def ok_result(power="71.25 W", memory="1024 MiB"):
    return {"returncode": 0, "timed_out": False,
            "stdout": f"{HEADER}\n0, {GPU0}, {power}, {memory}\n1, {GPU1}, 80.50 W, 2048 MiB\n"}


def body(*rows):
    return {"returncode": 0, "timed_out": False, "stdout": "\n".join((HEADER, *rows)) + "\n"}


def run(tmp_path, results, **changes):
    clock = FakeClock()
    runner = FakeRunner(clock, results)
    arguments = {"runner": runner, "clock": clock, "devices": [GPU0], "device_key": "uuid",
                 "clock_id": "guest-monotonic", "interval_s": 1, "duration_s": 3,
                 "timeout_s": 0.5, "scope": "synthetic"}
    arguments.update(changes)
    summary = sample(tmp_path / "out", **arguments)
    return summary, load_samples(tmp_path / "out" / "samples.jsonl"), runner, clock


def test_three_ticks_known_answer_and_exact_schema(tmp_path):
    summary, samples, runner, _ = run(tmp_path, [ok_result(), ok_result("72.0 W"),
                                                 ok_result("[N/A]")])
    assert [call[0] for call in runner.calls] == [list(QUERY_ARGV)] * 3
    # B1-F4: the recorded host invocation format (header and units), not an unrecorded one.
    assert QUERY_ARGV == ("nvidia-smi", "--query-gpu=index,uuid,power.draw,memory.used",
                          "--format=csv")
    assert [call[2] for call in runner.calls] == [10_000_000, 11_000_000, 12_000_000]
    assert samples[:2] == [
        {"metric": "nvidia_smi:power.draw", "device": GPU0, "clock_id": "guest-monotonic",
         "t_us": 10_000_000, "status": "ok", "value": 71.25, "raw_ref": f"{RAW_NAME}#tick=0:row=0"},
        {"metric": "nvidia_smi:memory.used", "device": GPU0, "clock_id": "guest-monotonic",
         "t_us": 10_000_000, "status": "ok", "value": 1024.0,
         "raw_ref": f"{RAW_NAME}#tick=0:row=0"},
    ]
    assert samples[4] == {"metric": "nvidia_smi:power.draw", "device": GPU0,
                          "clock_id": "guest-monotonic", "t_us": 12_000_000,
                          "status": "not_supported", "value": None,
                          "raw_ref": f"{RAW_NAME}#tick=2:row=0"}
    assert summary["ticks"] == 3 and summary["missed_ticks"] == 0
    assert summary["sample_count"] == 6 and summary["unallocated_rows"] == 3
    assert summary["status_counts"]["ok"] == 5 and summary["status_counts"]["not_supported"] == 1
    assert summary["clock_metadata"] == {"clock_id": "guest-monotonic", "clock_kind": "monotonic",
                                         "clock_scope": "host", "utc_anchor": None}
    saved = json.loads((tmp_path / "out" / SUMMARY_NAME).read_text())
    assert saved == summary
    raw = [json.loads(line) for line in (tmp_path / "out" / RAW_NAME).read_text().splitlines()]
    assert raw[0]["tick"] == 0 and raw[0]["returncode"] == 0 and GPU0 in raw[0]["stdout"]


def test_samples_are_accepted_by_the_telemetry_summary(tmp_path):
    _, samples, _, _ = run(tmp_path, [ok_result(), ok_result(), ok_result()])
    allocation = {"allocation_id": "a", "clock_id": "guest-monotonic", "gpu_count": 1,
                  "devices": [GPU0], "billed_start_us": 10_000_000, "billed_end_us": 13_000_000,
                  "segments": [{"phase": "active", "start_us": 10_000_000,
                                "end_us": 13_000_000}]}
    clock = {"clock_id": "guest-monotonic", "clock_kind": "monotonic", "clock_scope": "host",
             "utc_anchor": None}
    report = summarize_telemetry(samples, allocation, [], clock_metadata=clock,
                                 max_sample_gap_us=2_000_000, scope="synthetic")
    codes = {item["code"] for item in report["issues"]}
    assert not codes & {"sample_invalid", "metric_not_pinned", "device_not_allocated",
                        "clock_scope_mismatch", "duplicate_sample", "all_samples_dropped"}


@pytest.mark.parametrize("result, status", [
    ({"returncode": 1, "stdout": "", "timed_out": False}, "scrape_failed"),
    ({"returncode": None, "stdout": "", "timed_out": True}, "scrape_failed"),
    ({"returncode": 0, "stdout": "garbage\n", "timed_out": False}, "scrape_failed"),
    (body(f"0, {GPU0}, 1 W, 2 MiB", f"0, {GPU0}, 1 W, 2 MiB"), "scrape_failed"),
    (body(f"1, {GPU1}, 1 W, 2 MiB"), "no_data"),
    (body(f"0, {GPU0}, [Insufficient Permissions], 2 MiB"), "no_permission"),
    (body(f"0, {GPU0}, [Unknown Error], 2 MiB"), "scrape_failed"),
    (body(f"0, {GPU0}, nan W, 2 MiB"), "scrape_failed"),
    # Wrong or missing unit for the field, and the never-recorded nounits form, fail closed.
    (body(f"0, {GPU0}, 12 MiB, 2 MiB"), "scrape_failed"),
    (body(f"0, {GPU0}, 12, 2 MiB"), "scrape_failed"),
    # A missing or different header line makes the whole scrape untrustworthy.
    ({"returncode": 0, "stdout": f"0, {GPU0}, 12 W, 2 MiB\n", "timed_out": False},
     "scrape_failed"),
    ({"returncode": 0, "stdout": f"index, uuid, power.draw, memory.used\n0, {GPU0}, 12 W, 2 MiB\n",
      "timed_out": False}, "scrape_failed"),
])
def test_failures_are_explicit_never_zero(result, status):
    samples, _ = tick_samples(result, tick=4, t_us=5, devices=[GPU0], device_key="uuid",
                              clock_id="c")
    power = samples[0]
    assert power["metric"] == "nvidia_smi:power.draw"
    assert power["status"] == status and power["value"] is None


def test_runner_exception_is_a_scrape_failure(tmp_path):
    summary, samples, _, _ = run(tmp_path, [RuntimeError("boom"), ok_result(), ok_result()])
    assert summary["runner_errors"] == 1
    assert {s["status"] for s in samples if s["t_us"] == 10_000_000} == {"scrape_failed"}


def test_overrun_skips_missed_ticks_without_backfill(tmp_path):
    slow = {**ok_result(), "cost_us": 2_500_000}
    summary, samples, runner, _ = run(tmp_path, [slow, ok_result()], duration_s=5)
    # Tick 0 at 10.0 s runs until 12.5 s, so ticks 1 and 2 are skipped; tick 3 at 13.0 s.
    assert [call[2] for call in runner.calls] == [10_000_000, 13_000_000, 14_000_000]
    assert summary["missed_ticks"] == 2 and summary["ticks"] == 3
    assert summary["issues"][0]["code"] == "missed_ticks"
    assert sorted({s["t_us"] for s in samples}) == [10_000_000, 13_000_000, 14_000_000]


def test_index_device_key_and_should_stop(tmp_path):
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2

    summary, samples, _, _ = run(tmp_path, [], devices=["0", "1"], device_key="index",
                                 duration_s=10, should_stop=stop)
    assert summary["ticks"] == 2
    assert {s["device"] for s in samples} == {"0", "1"}
    assert summary["unallocated_rows"] == 0


def test_parse_known_answer():
    rows = parse_query_output(f"{HEADER}\n0, {GPU0}, 71.25 W, 1024 MiB\n", device_key="uuid")
    assert rows == {GPU0: (0, {"index": "0", "uuid": GPU0, "power.draw": "71.25 W",
                               "memory.used": "1024 MiB"})}
    with pytest.raises(ValueError):
        parse_query_output(f"{HEADER}\n0, a, 1\n", device_key="uuid")


# Literal lines recorded on the 2026-11-01 host (evidence/s2-vast-host-20261101-v1):
RECORDED_TELEMETRY = ("index, temperature.gpu, power.draw [W], clocks.current.sm [MHz]\n"
                      "0, 33, 18.15 W, 180 MHz\n")  # probe-v2/gpu_telemetry.log
RECORDED_NVML = ("name, driver_version, power.draw [W], temperature.gpu, "
                 "clocks.current.sm [MHz], memory.used [MiB]\n"
                 "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 580.95.05, 42.72 W, 36, "
                 "180 MHz, 0 MiB\n")  # monitor-v1/nvml-after.log
RECORDED_IDENTITY = ("index, name, uuid, driver_version, memory.total [MiB], pci.bus_id\n"
                     "0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, "
                     "GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57, 580.95.05, 97887 MiB, "
                     "00000000:00:07.0\n")  # probe-v2/gpu_identity.log


def recorded_cells(text):
    header, row = (line.split(", ") for line in text.splitlines())
    return dict(zip(header, row, strict=True))


def test_recorded_host_cells_parse_to_known_answers():
    """B1-F4: the recorded units format is accepted cell by cell, with its own header names."""
    telemetry, nvml = recorded_cells(RECORDED_TELEMETRY), recorded_cells(RECORDED_NVML)
    identity = recorded_cells(RECORDED_IDENTITY)
    gpu = identity["uuid"]
    stdout = (f"{HEADER}\n{identity['index']}, {gpu}, {telemetry['power.draw [W]']}, "
              f"{nvml['memory.used [MiB]']}\n")
    samples, extra = tick_samples({"returncode": 0, "timed_out": False, "stdout": stdout},
                                  tick=0, t_us=1, devices=[gpu], device_key="uuid",
                                  clock_id="c")
    assert [(s["metric"], s["status"], s["value"]) for s in samples] == [
        ("nvidia_smi:power.draw", "ok", 18.15), ("nvidia_smi:memory.used", "ok", 0.0)]
    assert extra == 0
    assert gpu == "GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57"
    samples, _ = tick_samples({"returncode": 0, "timed_out": False,
                               "stdout": stdout.replace("18.15 W", "42.72 W")},
                              tick=0, t_us=1, devices=[gpu], device_key="uuid", clock_id="c")
    assert samples[0]["value"] == 42.72


def test_query_format_is_labelled_qualify_on_host(tmp_path):
    summary, _, _, _ = run(tmp_path, [])
    assert summary["query_format"] == "qualify_on_host"


def test_all_scrape_failures_are_not_a_pass(tmp_path):
    """B1-F10: ticks that produced no ok sample never make the sampler 'passed'."""
    failing = [{"returncode": 1, "stdout": "", "timed_out": False} for _ in range(3)]
    summary, samples, _, _ = run(tmp_path, failing)
    assert {s["status"] for s in samples} == {"scrape_failed"}
    assert summary["status"] == "unverified"
    issue = next(i for i in summary["issues"] if i["code"] == "no_ok_samples")
    assert issue["severity"] == "blocking"
    assert issue["message"].endswith(
        f"[['{GPU0}', 'nvidia_smi:memory.used'], ['{GPU0}', 'nvidia_smi:power.draw']]")


def test_one_metric_never_ok_is_not_a_pass(tmp_path):
    """B1-F10: every declared device x metric needs at least one ok sample."""
    results = [ok_result("[N/A]") for _ in range(3)]
    summary, _, _, _ = run(tmp_path, results)
    assert summary["status"] == "unverified"
    assert summary["issues"][-1]["message"].endswith(f"[['{GPU0}', 'nvidia_smi:power.draw']]")
    summary, _, _, _ = run(tmp_path / "second", [ok_result("[N/A]"), ok_result(), ok_result()])
    assert summary["status"] == "passed" and summary["issues"] == []


@pytest.mark.parametrize("changes", [
    {"devices": []}, {"devices": [GPU0, GPU0]}, {"device_key": "name"},
    {"clock_id": ""}, {"interval_s": 0}, {"timeout_s": 1}, {"duration_s": float("inf")},
    {"scope": None}, {"utc_anchor": " "},
])
def test_invalid_inputs_fail_before_sampling(tmp_path, changes):
    with pytest.raises(ValueError):
        run(tmp_path, [], **changes)
    assert not (tmp_path / "out").exists()


def test_existing_output_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path, [])


def test_cli_uses_injected_runner_and_clock(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, [])
    code = main([str(tmp_path / "cli"), "--scope", "synthetic", "--device-key", "uuid",
                 "--device", GPU0, "--clock-id", "c", "--interval-s", "1", "--duration-s", "2",
                 "--timeout-s", "0.5"], runner=runner, clock=clock)
    assert code == 0 and len(runner.calls) == 2
