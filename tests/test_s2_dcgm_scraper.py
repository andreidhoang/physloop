"""Fake-runner known answers for the in-run dcgm-exporter scraper (F6 residual (a)).

No test runs curl, docker or a network call: every scrape goes through an injected fake
runner and clock. Bodies are the recorded scrapes copied byte-identically into
tests/fixtures/s2_telemetry_sources (provenance.json): power 52.306 W then 84.117 W,
energy 20078 then 105071 mJ (the known answer gate.s2_telemetry_sources already pins).

Hand-derived schedule (interval 1 s, duration 4 s, clock start 10 s, each scrape 0.1 s):
ticks at 10, 11, 12, 13 s; t_us is read immediately before each request.
"""

import hashlib
import json
from pathlib import Path

import pytest

from gate import s2_telemetry_sources as src
from scripts.s2_dcgm_scraper import (
    ENDPOINT,
    RAW_NAME,
    SCRAPES_NAME,
    SUMMARY_NAME,
    load_scrapes,
    main,
    scrape,
    scrape_argv,
    tick_record,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "s2_telemetry_sources"
S = 1_000_000
UUID0 = "GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57"
POWER = "DCGM_FI_DEV_POWER_USAGE"
ENERGY = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"


def fixture_bytes(name):
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    entry = next(item for item in provenance["sources"] if item["file"] == name)
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return data


class FakeClock:
    def __init__(self, start=10 * S):
        self.now = start
        self.sleeps = []

    def monotonic_us(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += round(seconds * S)


class FakeRunner:
    """Scripted results; each call advances the clock by the scrape's duration."""

    def __init__(self, clock, results, cost_us=100_000):
        self.clock, self.results, self.cost_us, self.calls = clock, list(results), cost_us, []

    def __call__(self, argv, timeout_s):
        self.calls.append((list(argv), timeout_s, self.clock.now))
        result = self.results.pop(0) if self.results else ok(b"")
        cost = self.cost_us
        if isinstance(result, dict) and "cost_us" in result:
            result = dict(result)
            cost = result.pop("cost_us")
        self.clock.now += cost
        if isinstance(result, Exception):
            raise result
        return result


def ok(body):
    return {"returncode": 0, "stdout": body, "timed_out": False}


def run(tmp_path, results, **changes):
    clock = FakeClock()
    runner = FakeRunner(clock, results)
    arguments = {"runner": runner, "clock": clock, "clock_id": "guest-mono-1",
                 "interval_s": 1, "duration_s": 4, "timeout_s": 0.5, "scope": "synthetic"}
    arguments.update(changes)
    summary = scrape(tmp_path / "out", **arguments)
    return summary, load_scrapes(tmp_path / "out" / SCRAPES_NAME), runner, clock


def test_endpoint_and_argv_are_the_exporter_control_scripts_own():
    # scripts/vast_exporter_control.sh:128-131 starts the exporter on -a 127.0.0.1:9400;
    # :135 and :154 scrape it with curl --fail --silent --show-error --max-time.
    control = (ROOT / "scripts" / "vast_exporter_control.sh").read_text()
    assert ENDPOINT == "http://127.0.0.1:9400/metrics"
    assert '"$image" -a 127.0.0.1:9400 -c 1000' in control
    assert control.count(f"curl --fail --silent --show-error --max-time 2 {ENDPOINT}") == 2
    assert scrape_argv(2) == ["curl", "--disable", "--fail", "--silent", "--show-error",
                              "--max-time", "2", ENDPOINT]
    assert scrape_argv(0.5) == ["curl", "--disable", "--fail", "--silent", "--show-error",
                                "--max-time", "0.5", ENDPOINT]


def test_four_ticks_known_answer_feed_parse_dcgm_scrapes(tmp_path):
    first, second = fixture_bytes("sample-0.prom"), fixture_bytes("sample-1.prom")
    summary, records, runner, _ = run(tmp_path, [
        ok(first), ok(second),
        {"returncode": None, "stdout": b"", "timed_out": True},
        {"returncode": 7, "stdout": b"", "timed_out": False}])
    assert [call[0] for call in runner.calls] == [scrape_argv(0.5)] * 4
    assert [call[1] for call in runner.calls] == [0.5] * 4
    assert records == [
        {"t_us": 10 * S, "text": first.decode(), "ref": f"{RAW_NAME}#tick=0"},
        {"t_us": 11 * S, "text": second.decode(), "ref": f"{RAW_NAME}#tick=1"},
        {"t_us": 12 * S, "text": None, "ref": f"{RAW_NAME}#tick=2"},
        {"t_us": 13 * S, "text": None, "ref": f"{RAW_NAME}#tick=3"},
    ]
    assert summary["ticks"] == 4 and summary["missed_ticks"] == 0
    assert summary["captured"] == 2
    assert summary["failures_by_reason"] == {"nonzero_exit": 1, "timed_out": 1}
    assert summary["status"] == "unverified"
    assert [i["code"] for i in summary["issues"]] == ["scrape_failures"]

    parsed = src.parse_dcgm_scrapes(records, device_key="UUID", device_map={UUID0: "gpu0"},
                                    clock_id="guest-mono-1", scope="synthetic")
    assert parsed["failed_scrapes"] == [12 * S, 13 * S]
    rows = {(r["metric"], r["t_us"]): (r["status"], r["value"]) for r in parsed["samples"]}
    assert rows[(POWER, 10 * S)] == ("ok", 52.306)
    assert rows[(POWER, 11 * S)] == ("ok", 84.117)
    assert rows[(ENERGY, 10 * S)] == ("ok", 20078)
    assert rows[(ENERGY, 11 * S)] == ("ok", 105071)
    assert rows[(POWER, 12 * S)] == ("scrape_failed", None)
    assert parsed["input_hashes"]["scrape_texts_sha256"][:2] == [
        "742a199cb37ec3454de18986b9afc59bb0551b62992cccb39d1b04ec4cfce550",
        "627cc058037510d6835ff9d5180e5ec0a9c63a8f231e0743b422bf8983fbeddb"]
    assert parsed["input_hashes"]["scrape_texts_sha256"][2:] == [None, None]

    raw = [json.loads(line) for line in (tmp_path / "out" / RAW_NAME).read_text().splitlines()]
    assert raw[0] == {"tick": 0, "t_us": 10 * S, "returncode": 0, "timed_out": False,
                      "body_bytes": len(first),
                      "body_sha256": hashlib.sha256(first).hexdigest(), "failure": None}
    assert raw[2]["failure"] == "timed_out" and raw[3]["failure"] == "nonzero_exit"
    assert raw[3]["returncode"] == 7
    written = json.loads((tmp_path / "out" / SUMMARY_NAME).read_text())
    assert written == summary


def test_body_is_stored_unstripped_so_the_parser_sees_the_trailing_newline(tmp_path):
    body = fixture_bytes("sample-1.prom")
    assert body.endswith(b"\n")
    _, records, _, _ = run(tmp_path, [ok(body), ok(body.rstrip(b"\n"))], duration_s=2)
    assert records[0]["text"].encode() == body
    truncated = src.parse_dcgm_prom(records[1]["text"], scrape_t_us=records[1]["t_us"],
                                    device_key="UUID", device_map={UUID0: "gpu0"},
                                    clock_id="guest-mono-1", scrape_ref=records[1]["ref"],
                                    scope="synthetic")
    assert truncated["scrape"]["failure"] == "scrape_truncated"


@pytest.mark.parametrize("result, reason", [
    (ok(b"DCGM_FI_DEV_GPU_UTIL 1\n\xff\n"), "undecodable_body"),
    (ok("DCGM_FI_DEV_GPU_UTIL 1\n"), "invalid_runner_result"),
    ({"returncode": 0, "timed_out": False}, "invalid_runner_result"),
    ({"returncode": True, "stdout": b"x\n", "timed_out": False}, "invalid_runner_result"),
    (RuntimeError("boom"), "runner_error"),
    (None, "invalid_runner_result"),
])
def test_failed_or_unreadable_scrape_is_text_none_never_a_guess(tmp_path, result, reason):
    summary, records, _, _ = run(tmp_path, [result], duration_s=1)
    assert records == [{"t_us": 10 * S, "text": None, "ref": f"{RAW_NAME}#tick=0"}]
    assert summary["failures_by_reason"] == {reason: 1}
    assert summary["captured"] == 0
    assert summary["status"] == "unverified"
    assert "no_scrape_captured" in {i["code"] for i in summary["issues"]}


def test_empty_200_body_is_captured_as_empty_text_and_fails_in_the_parser(tmp_path):
    summary, records, _, _ = run(tmp_path, [ok(b"")], duration_s=1)
    assert records[0]["text"] == ""
    assert summary["captured"] == 1 and summary["status"] == "passed"
    parsed = src.parse_dcgm_scrapes(records, device_key="UUID", device_map={UUID0: "gpu0"},
                                    clock_id="guest-mono-1", scope="synthetic")
    assert parsed["failed_scrapes"] == [10 * S]


def test_overrun_skips_missed_ticks_and_never_backfills(tmp_path):
    slow = dict(ok(fixture_bytes("sample-0.prom")), cost_us=2_500_000)
    summary, records, _, _ = run(tmp_path, [slow, ok(fixture_bytes("sample-1.prom"))],
                                 duration_s=5)
    # Tick 0 at 10 s runs to 12.5 s; ticks 1 and 2 (11 s, 12 s) are skipped; tick 3 at 13 s;
    # tick 4 at 14 s.
    assert [r["t_us"] for r in records] == [10 * S, 13 * S, 14 * S]
    assert [r["ref"] for r in records] == [f"{RAW_NAME}#tick={k}" for k in (0, 3, 4)]
    assert summary["ticks"] == 3 and summary["missed_ticks"] == 2
    assert "missed_ticks" in {i["code"] for i in summary["issues"]}


def test_should_stop_and_zero_ticks_are_not_run(tmp_path):
    summary, records, runner, _ = run(tmp_path, [], should_stop=lambda: True)
    assert records == [] and runner.calls == []
    assert summary["status"] == "not_run" and summary["ticks"] == 0


def test_clock_that_does_not_increase_blocks_and_stops(tmp_path):
    class StuckClock(FakeClock):
        def sleep(self, seconds):
            self.sleeps.append(seconds)  # time never advances

    clock = StuckClock()
    runner = FakeRunner(clock, [ok(fixture_bytes("sample-0.prom"))] * 3, cost_us=0)
    summary = scrape(tmp_path / "out", runner=runner, clock=clock, clock_id="c",
                     interval_s=1, duration_s=3, timeout_s=0.5, scope="synthetic")
    records = load_scrapes(tmp_path / "out" / SCRAPES_NAME)
    assert [r["t_us"] for r in records] == [10 * S]
    assert summary["status"] == "blocked"
    assert "clock_not_increasing" in {i["code"] for i in summary["issues"]}
    # parse_dcgm_scrapes requires strictly increasing times; the file stays consumable.
    src.parse_dcgm_scrapes(records, device_key="UUID", device_map={UUID0: "gpu0"},
                           clock_id="c", scope="synthetic")


@pytest.mark.parametrize("changes", [
    {"timeout_s": 1}, {"timeout_s": 2}, {"interval_s": 0}, {"duration_s": -1},
    {"interval_s": float("nan")}, {"scope": "real"}, {"clock_id": " "}, {"clock_id": None},
    {"utc_anchor": ""},
    # G4-R4: the loop runs in whole microseconds; values that round to an empty interval or
    # duration, or a timeout that rounds up to the interval, are refused before mkdir.
    {"interval_s": 4e-7, "timeout_s": 1e-7}, {"duration_s": 4e-7},
    {"interval_s": 1.4e-6, "timeout_s": 1.2e-6},
])
def test_bad_inputs_are_refused_before_anything_runs(tmp_path, changes):
    with pytest.raises(ValueError):
        run(tmp_path, [], **changes)
    assert not (tmp_path / "out").exists()


def test_existing_output_directory_is_never_reused(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path, [])


def test_tick_record_known_answer():
    record, raw = tick_record(ok(b"a\n"), tick=5, t_us=7)
    assert record == {"t_us": 7, "text": "a\n", "ref": f"{RAW_NAME}#tick=5"}
    assert raw == {"tick": 5, "t_us": 7, "returncode": 0, "timed_out": False, "body_bytes": 2,
                   "body_sha256": hashlib.sha256(b"a\n").hexdigest(), "failure": None}


def test_summary_carries_the_shared_envelope_and_is_deterministic(tmp_path):
    bodies = [ok(fixture_bytes("sample-0.prom")), ok(fixture_bytes("sample-1.prom"))]
    first, _, _, _ = run(tmp_path / "a", list(bodies), duration_s=2)
    second, _, _, _ = run(tmp_path / "b", list(bodies), duration_s=2)
    assert first == second
    for key in ("artifact_kind", "schema_version", "scope", "producer", "input_hashes",
                "units", "status", "issues"):
        assert key in first
    assert first["artifact_kind"] == "s2_dcgm_scrape_capture"
    assert first["status"] == "passed"
    assert first["endpoint"] == ENDPOINT
    assert first["parse_with"] == "gate.s2_telemetry_sources.parse_dcgm_scrapes"
    assert first["host_qualification"] == "qualify_on_host"
    assert first["clock_metadata"] == {"clock_id": "guest-mono-1", "clock_kind": "monotonic",
                                       "clock_scope": "host", "utc_anchor": None}
    assert first["input_hashes"]["scrapes_jsonl"] == hashlib.sha256(
        (tmp_path / "a" / "out" / SCRAPES_NAME).read_bytes()).hexdigest()
    json.dumps(first, allow_nan=False)


def test_cli_with_injected_runner(tmp_path):
    clock = FakeClock()
    runner = FakeRunner(clock, [ok(fixture_bytes("sample-1.prom"))])
    code = main([str(tmp_path / "cli"), "--scope", "synthetic", "--clock-id", "guest-mono-1",
                 "--interval-s", "1", "--duration-s", "1", "--timeout-s", "0.5"],
                runner=runner, clock=clock)
    assert code == 0
    assert len(load_scrapes(tmp_path / "cli" / SCRAPES_NAME)) == 1
    runner = FakeRunner(clock, [{"returncode": 7, "stdout": b"", "timed_out": False}])
    code = main([str(tmp_path / "cli2"), "--scope", "synthetic", "--clock-id", "c",
                 "--interval-s", "1", "--duration-s", "1", "--timeout-s", "0.5"],
                runner=runner, clock=clock)
    assert code == 1
