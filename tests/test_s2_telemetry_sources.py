"""Known answers and negative controls for gate.s2_telemetry_sources (CPU-07/S2.7 producers).

Hand-derived oracles (seconds on the guest monotonic clock ``guest-mono-1``; the anchor
maps 2026-11-01T07:00:00Z to 1,000 s; billed time is [1,000 s, 2,000 s), 1 GPU):

- Lifecycle: provisioning from billed start to bootstrapping (+100 s), bootstrap to
  running (+300 s), running to exporting (+800 s), export to stopping (+860 s), teardown
  to billed end (absence verified at +1,000 s).
- Scenario 1 (no unknown time): job-a useful 1310-1400, job-b failed 1410-1450, job-c
  useful 1460-1700, so provisioning 100, bootstrap 200, unit_startup 10, useful 90 + 240 =
  330, idle 10 + 10 + 100 = 120, retry_or_failed 40, export 60, teardown 140; sum 1,000.
  Two useful episodes over 1,000/3,600 h = 36/5 per wall-hour.
- Scenario 2 (2 s clock-alignment uncertainty, a retried unit, an interrupted unit): bands
  of +-2 s around the five billed lifecycle phase changes (4 s each, the last clipped to
  2 s) are 18 s unknown; the retried unit 1410-1500 is 90 s unknown (its completed retry
  awaits Huy's retry-selection rule); the recorded interruption 1510-1600 is
  retry_or_failed; unknown total 108 s, useful 90 s.
- Scenario 3 (hard-killed unit, no terminal event): 1410-1800 is 390 s unknown.
- DCGM values are the recorded scrapes copied byte-identically into
  tests/fixtures/s2_telemetry_sources (provenance.json): power 52.306 W then 84.117 W,
  energy 20078 then 105071 mJ, so the counter lower bound is 84993 mJ = 84993/1000 J.
- Counter 1000, 3000, (failed scrape), 500 mJ: one reset at 3 s, lower-bound increase
  2000 + 500 = 2500 mJ.
"""

import copy
import hashlib
import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gate import s2_telemetry as tel
from gate import s2_telemetry_sources as src
from gate.s2_execution import LifecycleJournal
from gate.s2_ledger import reconcile_attempts

S = 1_000_000
FIXTURES = Path(__file__).parent / "fixtures" / "s2_telemetry_sources"
CLOCK_ID = "guest-mono-1"
CLOCK = {"clock_id": CLOCK_ID, "clock_kind": "monotonic", "clock_scope": "host",
         "utc_anchor": "2026-11-01T07:00:00+00:00"}
UUID0 = "GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57"
POWER = "DCGM_FI_DEV_POWER_USAGE"
ENERGY = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
FB = "DCGM_FI_DEV_FB_USED"
UTIL = "DCGM_FI_DEV_GPU_UTIL"
PCIE = "DCGM_FI_DEV_PCIE_REPLAY_COUNTER"
XID = "DCGM_FI_DEV_XID_ERRORS"
# dcgm-exporter 4.8.4 etc/default-counters.csv (tests/fixtures/s2_telemetry_sources/dcgm-exporter).
XID_HELP = "Value of the last XID error encountered."
CSV_SHA256 = "ddf9d12eb7c004a24241a84221843ce178c5ca075ffdf711e2d46046e7000006"
# HELP/TYPE text exactly as recorded in evidence/s2-vast-host-20261101-v1/exporter-v1.
META = {
    POWER: ("Power draw (in W).", "gauge"),
    ENERGY: ("Total energy consumption since the driver was last reloaded (in mJ).", "counter"),
    FB: ("Framebuffer memory used (in MiB).", "gauge"),
    UTIL: ("GPU utilization (in %).", "gauge"),
    PCIE: ("Total number of PCIe retries.", "counter"),
}
UNPINNED_RECORDED = [
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_DEC_UTIL", "DCGM_FI_DEV_ENC_UTIL",
    "DCGM_FI_DEV_FB_FREE", "DCGM_FI_DEV_FB_RESERVED", "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_MEMORY_TEMP", "DCGM_FI_DEV_MEM_CLOCK", "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_ROW_REMAP_FAILURE", "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_VGPU_LICENSE_STATUS",
    "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_PCIE_TX_BYTES", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
]


def fixture_text(name):
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    entry = next(item for item in provenance["sources"] if item["file"] == name)
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return data.decode()


def codes(report, severity=None):
    return {item["code"] for item in report["issues"]
            if severity is None or item["severity"] == severity}


def parse(text, t_s=0, device_map=None, device_key="gpu", ref="scrape-0.prom"):
    return src.parse_dcgm_prom(
        text, scrape_t_us=int(t_s * S), device_key=device_key,
        device_map=device_map if device_map is not None else {"0": "gpu0"},
        clock_id=CLOCK_ID, scrape_ref=ref, scope="synthetic")


def series(metric, value, gpu="0", extra=""):
    return f'{metric}{{gpu="{gpu}",UUID="GPU-{gpu}"{extra}}} {value}\n'


def block(metric, *lines, help_text=None, type_text=None):
    help_text = META[metric][0] if help_text is None else help_text
    type_text = META[metric][1] if type_text is None else type_text
    return f"# HELP {metric} {help_text}\n# TYPE {metric} {type_text}\n" + "".join(lines)


def by_key(report):
    return {(row["metric"], row["device"]): row for row in report["samples"]}


# ------------------------------------------------------------------ DCGM parser

def test_pins_match_the_recorded_exporter_help_and_type_lines():
    text = fixture_text("sample-1.prom")
    recorded = {}
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name, doc = line[7:].split(" ", 1)
            recorded.setdefault(name, {})["help"] = doc
        elif line.startswith("# TYPE "):
            name, kind = line[7:].split(" ", 1)
            recorded.setdefault(name, {})["type"] = kind
    for metric, (help_text, kind) in META.items():
        assert recorded[metric] == {"help": help_text, "type": kind}
        assert src.DCGM_PINNED[metric] == {"help": help_text, "type": kind}
    # Every DCGM metric gate.s2_telemetry pins is pinned here, with an agreeing kind.
    dcgm = {m for m in tel.PINNED_METRICS if m.startswith("DCGM_")}
    assert set(src.DCGM_PINNED) == dcgm
    kinds = {"gauge": "gauge", "counter": "counter", "event": "gauge"}
    for metric in dcgm:
        assert src.DCGM_PINNED[metric]["type"] == kinds[tel.PINNED_METRICS[metric][1]]
    # XID was never recorded; its HELP/TYPE come from the vendored exporter 4.8.4 CSV.
    assert src.DCGM_PINNED[XID] == {"help": XID_HELP, "type": "gauge"}
    assert exporter_csv()[XID] == {"help": XID_HELP, "type": "gauge"}


def exporter_csv():
    """Active rows of the vendored, hash-checked dcgm-exporter 4.8.4 default counters CSV."""
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    (entry,) = provenance["vendored"]
    data = (FIXTURES / entry["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"] == CSV_SHA256
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
    assert len(data) == entry["size_bytes"]
    assert entry["commit"] in entry["source_url"] and entry["tag"] == "4.8.4"
    rows = {}
    for line in data.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, kind, help_text = (part.strip() for part in line.split(",", 2))
        assert name not in rows
        rows[name] = {"help": help_text, "type": kind}
    return rows


def test_vendored_exporter_csv_and_license_are_byte_identical_upstream_copies():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    (entry,) = provenance["vendored"]
    license_bytes = (FIXTURES / entry["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == entry["license_file_sha256"]
    blob = b"blob " + str(len(license_bytes)).encode() + b"\0" + license_bytes
    assert hashlib.sha1(blob).hexdigest() == entry["license_file_git_blob_sha"]
    assert b"Apache License" in license_bytes and entry["license"] == "Apache-2.0"
    assert "Version=4.8.4" in entry["release_reason"]
    assert len(exporter_csv()) == 26  # every non-comment row; names unique


def test_every_recorded_help_and_type_line_equals_the_vendored_exporter_csv():
    # The evidence that the CSV is the image's own: each recorded HELP/TYPE pair (pinned and
    # unpinned metrics alike) equals the CSV's trimmed help and type columns.
    csv_rows = exporter_csv()
    for name in ("sample-0.prom", "sample-1.prom"):
        recorded = {}
        for line in fixture_text(name).splitlines():
            for keyword, field in (("# HELP ", "help"), ("# TYPE ", "type")):
                if line.startswith(keyword):
                    metric, text = line[len(keyword):].split(" ", 1)
                    recorded.setdefault(metric, {})[field] = text
        assert len(recorded) == {"sample-0.prom": 18, "sample-1.prom": 23}[name]
        for metric, pair in recorded.items():
            assert pair == csv_rows[metric], metric
    # Every DCGM metric this module pins carries exactly the CSV's HELP and TYPE.
    for metric, pin in src.DCGM_PINNED.items():
        assert pin == csv_rows[metric], metric
    # Active CSV rows the recorded host never exposed (no NVLink, no XID event, a label).
    exposed = {line.split(" ")[2] for line in fixture_text("sample-1.prom").splitlines()
               if line.startswith("# TYPE ")}
    assert sorted(set(csv_rows) - exposed) == [
        "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", XID, "DCGM_FI_DRIVER_VERSION"]


def test_recorded_scrape_known_answer():
    report = parse(fixture_text("sample-1.prom"), t_s=1501, device_key="UUID",
                   device_map={UUID0: "gpu0"}, ref="exporter-v1/sample-1.prom")
    assert report["artifact_kind"] == "s2_dcgm_scrape"
    assert report["scope"] == "synthetic"
    assert report["status"] == "passed"
    assert report["samples"] == [
        {"metric": FB, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 1501 * S,
         "status": "ok", "value": 0, "raw_ref": "exporter-v1/sample-1.prom#line=18"},
        {"metric": UTIL, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 1501 * S,
         "status": "ok", "value": 0, "raw_ref": "exporter-v1/sample-1.prom#line=24"},
        {"metric": PCIE, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 1501 * S,
         "status": "ok", "value": 0, "raw_ref": "exporter-v1/sample-1.prom#line=36"},
        {"metric": POWER, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 1501 * S,
         "status": "ok", "value": 84.117, "raw_ref": "exporter-v1/sample-1.prom#line=39"},
        {"metric": ENERGY, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 1501 * S,
         "status": "ok", "value": 105071, "raw_ref": "exporter-v1/sample-1.prom#line=48"},
    ]
    # XID is not in the exporter's counter set: reported, never a zero reading.
    assert report["pinned_absent"] == [XID]
    assert codes(report) == {"pinned_metric_absent"}
    assert [row["metric"] for row in report["unpinned_metrics"]] == UNPINNED_RECORDED
    assert all(row["series"] == 1 for row in report["unpinned_metrics"])
    assert report["unpinned_metrics"][0]["type"] == "counter"
    assert report["input_hashes"]["text_sha256"] == (
        "627cc058037510d6835ff9d5180e5ec0a9c63a8f231e0743b422bf8983fbeddb")
    assert report["unmapped_devices"] == []


def test_two_gpus_nan_inf_and_missing_device():
    text = (block(POWER, series(POWER, "100.5", "0"), series(POWER, "NaN", "1"))
            + block(ENERGY, series(ENERGY, "5000", "0"), series(ENERGY, "+Inf", "1"))
            + block(FB, series(FB, "1024", "0")))
    report = parse(text, t_s=7, device_map={"0": "gpu0", "1": "gpu1"})
    rows = {key: (row["status"], row["value"]) for key, row in by_key(report).items()}
    assert rows == {
        (FB, "gpu0"): ("ok", 1024), (FB, "gpu1"): ("no_data", None),
        (POWER, "gpu0"): ("ok", 100.5), (POWER, "gpu1"): ("no_data", None),
        (ENERGY, "gpu0"): ("ok", 5000), (ENERGY, "gpu1"): ("scrape_failed", None),
    }
    assert [(r["metric"], r["device"]) for r in report["samples"]] == [
        (FB, "gpu0"), (FB, "gpu1"), (POWER, "gpu0"), (POWER, "gpu1"),
        (ENERGY, "gpu0"), (ENERGY, "gpu1")]
    assert by_key(report)[(POWER, "gpu1")]["raw_ref"] == "scrape-0.prom#line=4"
    assert by_key(report)[(FB, "gpu1")]["raw_ref"] == "scrape-0.prom"
    assert report["pinned_absent"] == [UTIL, PCIE, XID]
    nonfinite = [item for item in report["issues"] if item["code"] == "non_finite_value"]
    assert sorted(item["severity"] for item in nonfinite) == ["error", "warning"]
    assert report["status"] == "blocked"  # +Inf is not a physical reading
    # Every sample is accepted by the consumer's exact sample schema.
    alloc = {"allocation_id": "a", "clock_id": CLOCK_ID, "gpu_count": 2,
             "devices": ["gpu0", "gpu1"], "billed_start_us": 0, "billed_end_us": 10 * S,
             "segments": [{"phase": "active", "start_us": 0, "end_us": 10 * S}]}
    summary = tel.summarize_telemetry(report["samples"], alloc, [], clock_metadata=CLOCK,
                                      max_sample_gap_us=2 * S, scope="synthetic")
    assert summary["samples"]["accepted"] == 6
    assert "sample_invalid" not in codes(summary)


def test_label_escapes_and_trailing_comma_parse():
    line = ('DCGM_FI_DEV_POWER_USAGE{gpu="0",modelName="A \\"quoted\\" \\\\ name\\n",} 10\n')
    report = parse(block(POWER, line))
    assert by_key(report)[(POWER, "gpu0")]["value"] == 10
    assert report["status"] == "passed"


def test_exporter_timestamp_is_ignored_with_a_warning():
    report = parse(block(POWER, series(POWER, "10 1700000000000")))
    assert by_key(report)[(POWER, "gpu0")]["value"] == 10
    assert by_key(report)[(POWER, "gpu0")]["t_us"] == 0
    assert "exporter_timestamp_ignored" in codes(report, "warning")


@pytest.mark.parametrize("text, code", [
    (None, "scrape_missing"),
    ("", "scrape_empty"),
    (block(POWER, series(POWER, "10")).rstrip("\n"), "scrape_truncated"),
    (block(POWER, 'DCGM_FI_DEV_POWER_USAGE{gpu="0" 10\n'), "malformed_line"),
    (block(POWER, series(POWER, "ten")), "malformed_line"),
    (block(POWER, 'DCGM_FI_DEV_POWER_USAGE{gpu="0",gpu="1"} 10\n'), "malformed_line"),
    (block(POWER, 'DCGM_FI_DEV_POWER_USAGE{gpu="\\q"} 10\n'), "malformed_line"),
    (block(POWER, series(POWER, "10")) + f"# TYPE {POWER} gauge\n", "duplicate_metadata"),
    (f"# TYPE {POWER} bogus\n" + series(POWER, "10"), "malformed_line"),
    ("\n", "scrape_no_samples"),
    ("\n\n  \n", "scrape_no_samples"),
    (f"# HELP {POWER} Power draw (in W).\n# TYPE {POWER} gauge\n", "scrape_no_samples"),
    ("go_goroutines 7\n", "scrape_no_samples"),
])
def test_failed_scrape_marks_every_pinned_series_scrape_failed(text, code):
    report = parse(text, t_s=3, device_map={"0": "gpu0", "1": "gpu1"})
    assert report["scrape"]["outcome"] == "failed"
    assert report["scrape"]["failure"] == code
    assert report["status"] == "blocked"
    assert len(report["samples"]) == 2 * len(src.DCGM_PINNED)
    assert {(row["status"], row["value"], row["raw_ref"]) for row in report["samples"]} == {
        ("scrape_failed", None, "scrape-0.prom")}
    assert code in codes(report, "error")


@pytest.mark.parametrize("mutate, code", [
    (lambda: block(POWER, series(POWER, "10"), help_text="Power draw (in mW)."),
     "pinned_help_mismatch"),
    (lambda: block(POWER, series(POWER, "10"), type_text="counter"), "pinned_type_mismatch"),
    (lambda: series(POWER, "10"), "pinned_metadata_missing"),
    (lambda: block(POWER, 'DCGM_FI_DEV_POWER_USAGE{UUID="GPU-0"} 10\n'),
     "device_label_missing"),
])
def test_untrusted_pinned_metric_is_scrape_failed_not_dropped(mutate, code):
    text = mutate() + block(FB, series(FB, "7"))
    report = parse(text)
    assert by_key(report)[(POWER, "gpu0")]["status"] == "scrape_failed"
    assert by_key(report)[(POWER, "gpu0")]["value"] is None
    assert by_key(report)[(FB, "gpu0")]["value"] == 7  # other metrics stay usable
    assert code in codes(report, "error")
    assert report["status"] == "blocked"


def test_duplicate_series_negative_counter_and_unmapped_device():
    text = (block(POWER, series(POWER, "10"), series(POWER, "11"), series(POWER, "12", "7"))
            + block(ENERGY, series(ENERGY, "-5")))
    report = parse(text)
    assert by_key(report)[(POWER, "gpu0")]["status"] == "scrape_failed"
    assert by_key(report)[(ENERGY, "gpu0")]["status"] == "scrape_failed"
    assert {"duplicate_series", "negative_counter_value", "unmapped_device"} <= codes(
        report, "error")
    assert report["unmapped_devices"] == ["7"]
    assert all(row["device"] == "gpu0" for row in report["samples"])


def test_xid_line_with_the_pinned_exporter_help_passes():
    text = f"# HELP {XID} {XID_HELP}\n# TYPE {XID} gauge\n" + series(XID, "79")
    report = parse(text)
    assert by_key(report)[(XID, "gpu0")] == {
        "metric": XID, "device": "gpu0", "clock_id": CLOCK_ID, "t_us": 0, "status": "ok",
        "value": 79, "raw_ref": "scrape-0.prom#line=3"}
    assert "help_not_pinned" not in codes(report)
    assert report["status"] == "passed"


def test_xid_line_with_other_help_is_a_pinned_help_mismatch():
    text = (f"# HELP {XID} synthetic stand-in help.\n# TYPE {XID} gauge\n"
            + series(XID, "79"))
    report = parse(text)
    assert by_key(report)[(XID, "gpu0")]["status"] == "scrape_failed"
    assert by_key(report)[(XID, "gpu0")]["value"] is None
    assert "pinned_help_mismatch" in codes(report, "error")
    assert report["status"] == "blocked"


def test_unpinned_help_still_leaves_the_scrape_unverified(monkeypatch):
    # The unpinned-HELP path stays: a pinned metric without recorded HELP text is
    # 'unverified', never 'passed'.
    monkeypatch.setitem(src.DCGM_PINNED, XID, {"help": None, "type": "gauge"})
    text = (f"# HELP {XID} synthetic stand-in help.\n# TYPE {XID} gauge\n"
            + series(XID, "79"))
    report = parse(text)
    assert by_key(report)[(XID, "gpu0")]["value"] == 79
    assert "help_not_pinned" in codes(report, "warning")
    assert report["status"] == "unverified"
    series_report = src.parse_dcgm_scrapes(
        [{"t_us": 0, "text": text, "ref": "s0"}], device_key="gpu",
        device_map={"0": "gpu0"}, clock_id=CLOCK_ID, scope="synthetic")
    assert series_report["status"] == "unverified"


@pytest.mark.parametrize("metric", [POWER, FB, UTIL])
def test_negative_gauge_is_scrape_failed(metric):
    report = parse(block(metric, series(metric, "-5")) + block(PCIE, series(PCIE, "0")))
    assert by_key(report)[(metric, "gpu0")]["status"] == "scrape_failed"
    assert by_key(report)[(metric, "gpu0")]["value"] is None
    assert by_key(report)[(PCIE, "gpu0")]["value"] == 0
    assert "negative_gauge_value" in codes(report, "error")
    assert report["status"] == "blocked"


def test_blank_scrape_inside_a_series_is_a_failed_scrape():
    scrapes = [{"t_us": 0, "text": energy_scrape("1000"), "ref": "s0"},
               {"t_us": 1 * S, "text": "\n", "ref": "s1"},
               {"t_us": 2 * S, "text": energy_scrape("2000"), "ref": "s2"}]
    report = src.parse_dcgm_scrapes(scrapes, device_key="gpu", device_map={"0": "gpu0"},
                                    clock_id=CLOCK_ID, scope="synthetic")
    assert report["failed_scrapes"] == [1 * S]
    assert report["status"] == "blocked"
    assert [(r["status"], r["value"]) for r in report["samples"]
            if r["metric"] == ENERGY and r["t_us"] == 1 * S] == [("scrape_failed", None)]


@pytest.mark.parametrize("kwargs", [
    {"device_key": "index"},
    {"device_key": None},
    {"device_map": {}},
    {"device_map": {"0": "gpu0", "1": "gpu0"}},
    {"device_map": {"0": ""}},
    {"device_map": [("0", "gpu0")]},
    {"scrape_t_us": True},
    {"scrape_t_us": -1},
    {"scrape_t_us": 1.5},
    {"clock_id": ""},
    {"scrape_ref": " "},
    {"scope": "real"},
    {"text": b"DCGM"},
])
def test_parser_rejects_bad_inputs(kwargs):
    base = {"text": "", "scrape_t_us": 0, "device_key": "gpu", "device_map": {"0": "gpu0"},
            "clock_id": CLOCK_ID, "scrape_ref": "r", "scope": "synthetic"}
    base.update(kwargs)
    text = base.pop("text")
    with pytest.raises(ValueError):
        src.parse_dcgm_prom(text, **base)


def test_parser_is_deterministic():
    text = fixture_text("sample-0.prom")
    first = parse(text, device_key="UUID", device_map={UUID0: "gpu0"})
    second = parse(text, device_key="UUID", device_map={UUID0: "gpu0"})
    assert first == second
    assert by_key(first)[(POWER, "gpu0")]["value"] == 52.306


def energy_scrape(value):
    return block(ENERGY, series(ENERGY, value)) + block(POWER, series(POWER, "50"))


def test_scrapes_counter_reset_and_scrape_gap_known_answer():
    scrapes = [
        {"t_us": 0, "text": energy_scrape("1000"), "ref": "s0.prom"},
        {"t_us": 1 * S, "text": energy_scrape("3000"), "ref": "s1.prom"},
        {"t_us": 2 * S, "text": None, "ref": "s2.prom"},  # curl failed: a scrape gap
        {"t_us": 3 * S, "text": energy_scrape("500"), "ref": "s3.prom"},
    ]
    report = src.parse_dcgm_scrapes(scrapes, device_key="gpu", device_map={"0": "gpu0"},
                                    clock_id=CLOCK_ID, scope="synthetic")
    assert report["artifact_kind"] == "s2_dcgm_scrapes"
    assert report["counter_resets"] == [
        {"metric": ENERGY, "device": "gpu0", "t_us": 3 * S, "previous": 3000, "value": 500}]
    assert report["failed_scrapes"] == [2 * S]
    assert [s["outcome"] for s in report["scrapes"]] == ["parsed", "parsed", "failed", "parsed"]
    energy = [(r["t_us"], r["status"], r["value"]) for r in report["samples"]
              if r["metric"] == ENERGY]
    assert energy == [(0, "ok", 1000), (1 * S, "ok", 3000), (2 * S, "scrape_failed", None),
                      (3 * S, "ok", 500)]
    assert report["status"] == "blocked"  # the failed scrape is an error issue
    alloc = {"allocation_id": "a", "clock_id": CLOCK_ID, "gpu_count": 1, "devices": ["gpu0"],
             "billed_start_us": 0, "billed_end_us": 3 * S,
             "segments": [{"phase": "active", "start_us": 0, "end_us": 3 * S}]}
    summary = tel.summarize_telemetry(report["samples"], alloc, [], clock_metadata=CLOCK,
                                      max_sample_gap_us=1_500_000, scope="synthetic")
    counter = summary["metrics"][ENERGY]["gpu0"]["counter"]
    assert counter["resets"] == 1
    assert counter["reset_events_us"] == [3 * S]
    assert counter["increase_lower_bound"]["exact"] == "2500"
    assert summary["metrics"][ENERGY]["gpu0"]["availability"] == "incomplete"
    assert summary["metrics"][ENERGY]["gpu0"]["status_counts"]["scrape_failed"] == 1
    assert summary["status"] == "passed"


def test_scrapes_require_strictly_increasing_times_and_exact_keys():
    good = {"t_us": 0, "text": energy_scrape("1"), "ref": "s0"}
    for scrapes in ([good, dict(good)], [dict(good, extra=1)], [{"t_us": 0, "text": ""}]):
        with pytest.raises(ValueError):
            src.parse_dcgm_scrapes(scrapes, device_key="gpu", device_map={"0": "gpu0"},
                                   clock_id=CLOCK_ID, scope="synthetic")


# ------------------------------------------------------------------ journals

ANCHOR_UTC = datetime(2026, 11, 1, 7, 0, 0, tzinfo=UTC)
ANCHOR = {"clock_id": CLOCK_ID, "monotonic_us": 1000 * S, "utc": "2026-11-01T07:00:00+00:00",
          "uncertainty_us": 0, "anchor_ref": "run-summary.json#clock_anchor"}
BILLED = {"allocation_id": "alloc-1", "clock_id": CLOCK_ID, "gpu_count": 1,
          "devices": ["gpu0"], "start_us": 1000 * S, "end_us": 2000 * S,
          "source_ref": "charges-after-delete.json"}


def transition(state):
    return {"type": "transition", "state": {"state": state}, "action": None,
            "observation": None, "issues": []}


NORMAL = [
    (-5, transition("admitted")), (-5, transition("create_intent")),
    (0, transition("create_requested")),
    (1, {"type": "observation", "observation": {"kind": "create_result"}}),
    (20, transition("bound")), (25, transition("armed")),
    (100, transition("bootstrapping")), (300, transition("running")),
    (300, transition("running")), (800, transition("exporting")),
    (850, transition("export_verified")), (860, transition("stopping")),
    (861, {"type": "effect", "operation": "stop", "resource_id": 1, "outcome": "requested"}),
    (900, transition("stopped")), (910, transition("destroying")),
    (1000, transition("absent_verified")),
]


def lifecycle(tmp_path, entries=NORMAL, name="lifecycle.jsonl"):
    times = iter([(ANCHOR_UTC + timedelta(seconds=rel)).isoformat() for rel, _ in entries])
    path = str(tmp_path / name)
    with LifecycleJournal(path, wall=lambda: next(times)) as journal:
        for _, payload in entries:
            journal.append(payload)
    return LifecycleJournal.read(path)


def job(name):
    return {"job_id": name, "phase": "synthetic", "arm": "A", "policy_id": "pol",
            "scene_id": f"scene-{name}", "road_id": f"road-{name}", "stratum": "s",
            "replicate_id": 0, "protocol_hash": "a" * 64, "requested_steps": 10}


def started(name, n, t, parent=None):
    return {"schema_version": 2, "event_id": f"{name}/attempt-{n}:started", "job_id": name,
            "attempt_id": f"{name}/attempt-{n}", "type": "started", "timestamp_us": t * S,
            "evidence_refs": [f"packet_unit:{name}"],
            "parent_attempt_id": None if parent is None else f"{name}/attempt-{parent}",
            "retry_reason": None if parent is None else "upstream_internal_retry",
            "upstream_identity": None}


REASONS = {"completed": "process_completed", "failed": "upstream_rollout_failed:retryable=False",
           "cancelled": "interrupted:KeyboardInterrupt"}


def terminal(name, n, kind, t, rollout=None):
    identity = None
    if kind == "completed":
        identity = {"run_uuid": "run-1", "clipgt_id": f"clip-{name}",
                    "rollout_id": rollout or f"roll-{name}-{n}"}
    return {"schema_version": 2, "event_id": f"{name}/attempt-{n}:{kind}", "job_id": name,
            "attempt_id": f"{name}/attempt-{n}", "type": kind, "timestamp_us": t * S,
            "evidence_refs": [f"unit_attempt_evidence:{name}"], "reason": REASONS[kind],
            "upstream_identity": identity}


def unit_record(index, name, events, scope="synthetic"):
    mine = [e for e in events if e["job_id"] == name]
    attempts = [{"attempt_id": e["attempt_id"], "type": e["type"], "reason": e["reason"]}
                for e in mine if e["type"] != "started"]
    return {"artifact_kind": "s2_unit_record", "schema_version": 1, "scope": scope,
            "producer": {"module": "scripts.s2_run_units", "function": "_Run.unit",
                         "source_sha256": "0" * 64},
            "input_hashes": {"packet": "1" * 64, "execution_plan": "2" * 64},
            "units": {"timeout_s": "seconds", "timestamp_us": "monotonic microseconds"},
            "status": "passed", "status_reasons": [], "index": index, "job_id": name,
            "scene_id": f"scene-{name}",
            "steps": [{"id": "budget_check", "status": "passed"},
                      {"id": "ledger_started", "status": "passed",
                       "attempt_id": f"{name}/attempt-1"},
                      {"id": "ledger_terminal", "status": "passed", "outcome": "recorded",
                       "attempts": attempts}],
            "outcome": "executed", "stop_run": None, "compose_exit_code": 0,
            "unit_outcome": None, "ledger_events": len(mine), "issues": []}


def scenario1_events():
    return [started("job-a", 1, 1310), terminal("job-a", 1, "completed", 1400),
            started("job-b", 1, 1410), terminal("job-b", 1, "failed", 1450),
            started("job-c", 1, 1460), terminal("job-c", 1, "completed", 1700)]


def scenario2_events():
    return [started("job-a", 1, 1310), terminal("job-a", 1, "completed", 1400),
            started("job-b", 1, 1410), terminal("job-b", 1, "failed", 1500),
            started("job-b", 2, 1500, parent=1), terminal("job-b", 2, "completed", 1500),
            started("job-c", 1, 1510), terminal("job-c", 1, "cancelled", 1600)]


def scenario3_events():
    return [started("job-a", 1, 1310), terminal("job-a", 1, "completed", 1400),
            started("job-d", 1, 1410)]


def ledger_and_rows(events, jobs=None, units=None, scope="synthetic"):
    names = jobs or sorted({e["job_id"] for e in events})
    report = reconcile_attempts([job(n) for n in names], events)
    if units is None:
        units = [unit_record(i, n, events, scope) for i, n in enumerate(names)]
    return report, src.useful_attempt_rows(report, units, clock_id=CLOCK_ID, scope=scope)


def partition(tmp_path, events, rows, *, anchor=ANCHOR, entries=NORMAL, billed=BILLED):
    return src.allocation_from_journals(
        lifecycle(tmp_path, entries), events, billed_interval=billed, clock_anchor=anchor,
        attempt_rows=rows, scope="synthetic")


def seconds(part):
    return [(seg["phase"], seg["start_us"] // S, seg["end_us"] // S) for seg in part["segments"]]


def assert_exact_partition(part):
    segs = part["segments"]
    assert segs[0]["start_us"] == part["billed_interval"]["start_us"]
    assert segs[-1]["end_us"] == part["billed_interval"]["end_us"]
    for left, right in itertools.pairwise(segs):
        assert left["end_us"] == right["start_us"]
    assert all(seg["start_us"] < seg["end_us"] for seg in segs)
    total = part["billed_interval"]["end_us"] - part["billed_interval"]["start_us"]
    assert sum(part["totals_us"].values()) == total
    assert sum(seg["end_us"] - seg["start_us"] for seg in segs) == total


def summarize(part, rows, samples=()):
    return tel.summarize_telemetry(list(samples), part["telemetry_allocation"], rows,
                                   clock_metadata=CLOCK, max_sample_gap_us=2 * S,
                                   scope="synthetic")


def test_scenario1_known_answer_passes_summarize_telemetry(tmp_path):
    events = scenario1_events()
    report, useful = ledger_and_rows(events)
    assert report["admitted"] is False  # job-b failed; outcome issue only
    assert useful["status"] == "passed"
    assert {k: v["useful"] for k, v in useful["classification"].items()} == {
        "job-a/attempt-1": True, "job-b/attempt-1": False, "job-c/attempt-1": True}
    assert useful["rows"][0] == {"attempt_id": "job-a/attempt-1", "job_id": "job-a",
                                 "clock_id": CLOCK_ID, "start_us": 1310 * S,
                                 "end_us": 1400 * S, "status": "completed", "useful": True}
    part = partition(tmp_path, events, useful["rows"])
    assert part["artifact_kind"] == "s2_allocation_partition"
    assert part["status"] == "passed"
    assert_exact_partition(part)
    assert seconds(part) == [
        ("provisioning", 1000, 1100), ("bootstrap", 1100, 1300), ("unit_startup", 1300, 1310),
        ("useful_rollout", 1310, 1400), ("idle", 1400, 1410), ("retry_or_failed", 1410, 1450),
        ("idle", 1450, 1460), ("useful_rollout", 1460, 1700), ("idle", 1700, 1800),
        ("export", 1800, 1860), ("teardown", 1860, 2000)]
    assert {k: v // S for k, v in part["totals_us"].items()} == {
        "provisioning": 100, "bootstrap": 200, "unit_startup": 10, "useful_rollout": 330,
        "retry_or_failed": 40, "idle": 120, "export": 60, "teardown": 140, "unknown": 0}
    assert part["unknown_intervals_us"] == []
    scrapes = src.parse_dcgm_scrapes(
        [{"t_us": 1500 * S, "text": fixture_text("sample-0.prom"), "ref": "sample-0.prom"},
         {"t_us": 1501 * S, "text": fixture_text("sample-1.prom"), "ref": "sample-1.prom"}],
        device_key="UUID", device_map={UUID0: "gpu0"}, clock_id=CLOCK_ID, scope="synthetic")
    assert scrapes["status"] == "passed"
    summary = summarize(part, useful["rows"], scrapes["samples"])
    assert summary["status"] == "passed"
    phases = {k: v["us"] // S for k, v in summary["allocation"]["partition"]["phases"].items()}
    assert phases == {"acquisition": 100, "startup": 200, "warm_up": 10, "active": 330,
                      "retry": 40, "idle": 120, "export": 60, "tail": 140}
    assert summary["useful_work"]["useful_episodes"] == 2
    assert summary["useful_work"]["per_wall_hour"]["exact"] == "36/5"
    assert summary["useful_work"]["status"] == "passed"
    counter = summary["energy"]["counter_delta"][ENERGY]["gpu0"]
    assert counter["lower_bound_energy_j"]["exact"] == "84993/1000"
    assert summary["xid"]["gpu0"]["availability"] == "absent"  # never zero XIDs


def test_scenario2_retry_interrupt_and_alignment_uncertainty(tmp_path):
    events = scenario2_events()
    _, useful = ledger_and_rows(events)
    got = {k: (v["useful"], v["reason"]) for k, v in useful["classification"].items()}
    assert got == {
        "job-a/attempt-1": (True, "established_completion"),
        "job-b/attempt-1": (False, "attempt_failed"),
        "job-b/attempt-2": (None, "retry_selection_unregistered"),
        "job-c/attempt-1": (False, "attempt_cancelled"),
    }
    assert useful["status"] == "unverified"
    anchor = dict(ANCHOR, uncertainty_us=2 * S)
    part = partition(tmp_path, events, useful["rows"], anchor=anchor)
    assert_exact_partition(part)
    assert seconds(part) == [
        ("provisioning", 1000, 1098), ("unknown", 1098, 1102), ("bootstrap", 1102, 1298),
        ("unknown", 1298, 1302), ("unit_startup", 1302, 1310), ("useful_rollout", 1310, 1400),
        ("idle", 1400, 1410), ("unknown", 1410, 1500), ("idle", 1500, 1510),
        ("retry_or_failed", 1510, 1600), ("idle", 1600, 1798), ("unknown", 1798, 1802),
        ("export", 1802, 1858), ("unknown", 1858, 1862), ("teardown", 1862, 1998),
        ("unknown", 1998, 2000)]
    reasons = {(seg["start_us"] // S): seg["reason"] for seg in part["segments"]
               if seg["phase"] == "unknown"}
    assert reasons == {1098: "clock_alignment_uncertainty", 1298: "clock_alignment_uncertainty",
                       1410: "unit_classification_unresolved",
                       1798: "clock_alignment_uncertainty", 1858: "clock_alignment_uncertainty",
                       1998: "clock_alignment_uncertainty"}
    assert part["totals_us"]["unknown"] == 108 * S
    assert part["totals_us"]["useful_rollout"] == 90 * S  # unknown never becomes useful
    assert part["status"] == "unverified"
    assert part["unknown_intervals_us"] == [[1098 * S, 1102 * S], [1298 * S, 1302 * S],
                                            [1410 * S, 1500 * S], [1798 * S, 1802 * S],
                                            [1858 * S, 1862 * S], [1998 * S, 2000 * S]]
    summary = summarize(part, useful["rows"])
    assert summary["status"] == "blocked"
    assert codes(summary, "error") == {"partition_gap"}
    partition_summary = summary["allocation"]["partition"]
    assert partition_summary["unaccounted_us"] == 108 * S
    assert partition_summary["phases"]["active"]["us"] == 90 * S
    assert summary["useful_work"]["useful_episodes"] == 1
    assert summary["useful_work"]["useful_unknown"] == 1
    assert summary["useful_work"]["status"] == "unverified"
    assert summary["useful_work"]["rates_are_lower_bounds"] is True


def test_scenario3_hard_killed_unit_is_unknown_until_the_next_known_boundary(tmp_path):
    events = scenario3_events()
    _, useful = ledger_and_rows(events)
    row = next(r for r in useful["rows"] if r["attempt_id"] == "job-d/attempt-1")
    assert row == {"attempt_id": "job-d/attempt-1", "job_id": "job-d", "clock_id": CLOCK_ID,
                   "start_us": 1410 * S, "end_us": None, "status": "unresolved",
                   "useful": None}
    assert useful["classification"]["job-d/attempt-1"]["reason"] == "attempt_incomplete"
    part = partition(tmp_path, events, useful["rows"])
    assert_exact_partition(part)
    assert seconds(part) == [
        ("provisioning", 1000, 1100), ("bootstrap", 1100, 1300), ("unit_startup", 1300, 1310),
        ("useful_rollout", 1310, 1400), ("idle", 1400, 1410), ("unknown", 1410, 1800),
        ("export", 1800, 1860), ("teardown", 1860, 2000)]
    assert part["segments"][5]["reason"] == "attempt_unresolved"
    assert part["segments"][5]["job_id"] == "job-d"
    summary = summarize(part, useful["rows"])
    assert summary["allocation"]["partition"]["unaccounted_us"] == 390 * S
    assert summary["useful_work"]["attempts_by_status"]["unresolved"] == 1
    assert "rollout_invalid" not in codes(summary)


def test_missing_clock_anchor_keeps_only_runner_units(tmp_path):
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    part = partition(tmp_path, events, useful["rows"], anchor=None)
    assert part["status"] == "blocked"
    assert "clock_anchor_missing" in codes(part, "error")
    assert_exact_partition(part)
    assert part["totals_us"]["useful_rollout"] == 330 * S
    assert part["totals_us"]["retry_or_failed"] == 40 * S
    assert part["totals_us"]["unknown"] == 630 * S
    for phase in ("provisioning", "bootstrap", "unit_startup", "idle", "export", "teardown"):
        assert part["totals_us"][phase] == 0


def test_runner_event_outside_running_phase_is_flagged_and_unknown(tmp_path):
    events = scenario1_events()
    events[0] = started("job-a", 1, 1250)  # inside bootstrap: clocks disagree
    _, useful = ledger_and_rows(events)
    part = partition(tmp_path, events, useful["rows"])
    assert "runner_outside_running_phase" in codes(part, "error")
    assert part["status"] == "blocked"
    assert_exact_partition(part)
    assert ("unknown", 1250, 1300) in seconds(part)
    assert ("useful_rollout", 1300, 1400) in seconds(part)
    assert part["totals_us"]["bootstrap"] == 150 * S


def test_running_without_runner_records_is_unknown(tmp_path):
    part = partition(tmp_path, [], [])
    assert_exact_partition(part)
    assert ("unknown", 1300, 1800) in seconds(part)
    assert part["totals_us"]["unknown"] == 500 * S
    assert "running_without_runner_records" in codes(part)


def test_lifecycle_time_regression_is_flagged(tmp_path):
    entries = list(NORMAL)
    entries[5] = (95, transition("armed"))
    entries[6] = (90, transition("bootstrapping"))  # wall clock steps back below armed
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    part = partition(tmp_path, events, useful["rows"], entries=entries)
    assert "lifecycle_time_regression" in codes(part, "error")
    assert ("unknown", 1090, 1095) in seconds(part)
    assert_exact_partition(part)


def test_tampered_lifecycle_chain_fails_closed(tmp_path):
    records = lifecycle(tmp_path)
    tampered = copy.deepcopy(records)
    tampered[6]["payload"]["state"]["state"] = "running"
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    for bad in (tampered, records[1:], [dict(records[0], extra=1)] + records[1:]):
        with pytest.raises(ValueError):
            src.allocation_from_journals(bad, events, billed_interval=BILLED,
                                         clock_anchor=ANCHOR, attempt_rows=useful["rows"],
                                         scope="synthetic")


def test_mismatched_inputs_fail_closed(tmp_path):
    records = lifecycle(tmp_path)
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    rows = useful["rows"]
    bad_rows = copy.deepcopy(rows)
    bad_rows[0]["end_us"] += 1
    cases = [
        {"billed_interval": dict(BILLED, clock_id="other")},
        {"clock_anchor": dict(ANCHOR, clock_id="other")},
        {"clock_anchor": dict(ANCHOR, utc="2026-11-01T07:00:00")},  # naive time
        {"clock_anchor": dict(ANCHOR, uncertainty_us=-1)},
        {"billed_interval": dict(BILLED, end_us=BILLED["start_us"])},
        {"billed_interval": dict(BILLED, devices=["gpu0", "gpu0"], gpu_count=2)},
        {"attempt_rows": bad_rows},
        {"attempt_rows": rows[1:]},
        {"attempt_rows": [dict(rows[0], clock_id="other")] + rows[1:]},
        {"runner_journal_records": events + [dict(events[0], timestamp_us=1)]},
        {"runner_journal_records": events[:1] + [{"type": "started"}]},
        {"scope": "real"},
    ]
    for case in cases:
        kwargs = {"billed_interval": BILLED, "clock_anchor": ANCHOR, "attempt_rows": rows,
                  "scope": "synthetic"}
        runner = case.pop("runner_journal_records", events)
        kwargs.update(case)
        with pytest.raises(ValueError):
            src.allocation_from_journals(records, runner, **kwargs)


def _mutated(rows, attempt_id, **changes):
    out = copy.deepcopy(rows)
    next(r for r in out if r["attempt_id"] == attempt_id).update(changes)
    return out


@pytest.mark.parametrize("attempt_id, changes", [
    ("job-b/attempt-1", {"useful": True}),                         # journal: failed
    ("job-b/attempt-1", {"useful": True, "status": "completed"}),  # journal: failed
    ("job-a/attempt-1", {"useful": False, "status": "cancelled"}),  # journal: completed
    ("job-a/attempt-1", {"useful": False}),                        # completed is never False
    ("job-b/attempt-1", {"status": "unresolved", "useful": None}),  # journal has a terminal
    ("job-b/attempt-1", {"status": "cancelled"}),                  # journal: failed
    ("job-a/attempt-1", {"start_us": 1310.0 * S}),                 # equal value, not an int
    ("job-a/attempt-1", {"end_us": 1400.0 * S}),
    ("job-a/attempt-1", {"end_us": None}),                         # completed needs an end
])
def test_partition_rejects_rows_that_contradict_the_runner_journal(tmp_path, attempt_id,
                                                                   changes):
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    with pytest.raises(ValueError):
        partition(tmp_path, events, _mutated(useful["rows"], attempt_id, **changes))


def test_partition_rejects_useful_completion_that_is_not_established(tmp_path):
    events = v1_events()  # v2 completions must be process_completed; v1 is not established
    _, useful = ledger_and_rows(events)
    assert useful["classification"]["job-a/attempt-1"]["reason"] == (
        "completion_not_established")
    part = partition(tmp_path, events, useful["rows"])  # the producer's own rows pass
    assert ("unknown", 1310, 1400) in seconds(part)
    with pytest.raises(ValueError):
        partition(tmp_path, events, _mutated(useful["rows"], "job-a/attempt-1", useful=True))


def test_partition_rejects_a_row_for_an_attempt_without_a_started_event(tmp_path):
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    runner = [e for e in events if e["event_id"] != "job-b/attempt-1:started"]
    with pytest.raises(ValueError):
        partition(tmp_path, runner, useful["rows"])


def test_partition_is_deterministic(tmp_path):
    events = scenario2_events()
    _, useful = ledger_and_rows(events)
    first = partition(tmp_path, events, useful["rows"], anchor=dict(ANCHOR, uncertainty_us=S))
    second = src.allocation_from_journals(
        LifecycleJournal.read(str(tmp_path / "lifecycle.jsonl")), copy.deepcopy(events),
        billed_interval=BILLED, clock_anchor=dict(ANCHOR, uncertainty_us=S),
        attempt_rows=copy.deepcopy(useful["rows"]), scope="synthetic")
    assert first == second


# ------------------------------------------------------------------ useful rows

def test_ledger_integrity_issue_blanks_every_useful_flag():
    events = scenario1_events()
    events[5] = terminal("job-c", 1, "completed", 1700, rollout="roll-job-a-1")
    events[5]["upstream_identity"]["clipgt_id"] = "clip-job-a"  # reused upstream identity
    report, useful = ledger_and_rows(events)
    assert "reused_upstream_identity" in {item["code"] for item in report["issues"]}
    assert useful["status"] == "blocked"
    assert "ledger_integrity_issue" in codes(useful, "error")
    assert {v["useful"] for v in useful["classification"].values()} == {None}
    alloc = {"allocation_id": "a", "clock_id": CLOCK_ID, "gpu_count": 1, "devices": ["gpu0"],
             "billed_start_us": 1000 * S, "billed_end_us": 2000 * S,
             "segments": [{"phase": "active", "start_us": 1000 * S, "end_us": 2000 * S}]}
    summary = tel.summarize_telemetry([], alloc, useful["rows"], clock_metadata=CLOCK,
                                      max_sample_gap_us=S, scope="synthetic")
    assert summary["useful_work"]["useful_unknown"] == 3
    assert summary["useful_work"]["useful_episodes"] == 0


def v1_events():
    return [{"event_id": "job-a/attempt-1:started", "job_id": "job-a",
               "attempt_id": "job-a/attempt-1", "type": "started", "timestamp_us": 1310 * S,
               "parent_attempt_id": None, "retry_reason": None, "upstream_identity": None},
              {"event_id": "job-a/attempt-1:completed", "job_id": "job-a",
               "attempt_id": "job-a/attempt-1", "type": "completed", "timestamp_us": 1400 * S,
               "reason": "exact_horizon",
               "upstream_identity": {"run_uuid": "r", "clipgt_id": "c", "rollout_id": "o"}}]


def test_v1_completion_is_not_established():
    events = v1_events()
    report = reconcile_attempts([job("job-a")], events)
    assert report["admitted"] is True
    useful = src.useful_attempt_rows(report, [unit_record(0, "job-a", events)],
                                     clock_id=CLOCK_ID, scope="synthetic")
    assert useful["classification"]["job-a/attempt-1"] == {
        "useful": None, "reason": "completion_not_established", "terminal_type": "completed",
        "terminal_reason": "exact_horizon", "unit_record": "present"}
    assert useful["status"] == "unverified"


def test_unit_record_cross_checks():
    events = scenario1_events()
    report = reconcile_attempts([job(n) for n in ("job-a", "job-b", "job-c")], events)
    units = [unit_record(i, n, events) for i, n in enumerate(("job-a", "job-b", "job-c"))]
    missing = src.useful_attempt_rows(report, units[1:], clock_id=CLOCK_ID, scope="synthetic")
    assert missing["classification"]["job-a/attempt-1"]["useful"] is None
    assert missing["classification"]["job-a/attempt-1"]["reason"] == "unit_record_missing"
    assert missing["classification"]["job-c/attempt-1"]["useful"] is True
    assert missing["status"] == "unverified"

    disagree = copy.deepcopy(units)
    disagree[0]["steps"][2]["attempts"][0]["type"] = "failed"
    result = src.useful_attempt_rows(report, disagree, clock_id=CLOCK_ID, scope="synthetic")
    assert result["classification"]["job-a/attempt-1"]["reason"] == "unit_record_disagrees"
    assert result["status"] == "blocked"

    counted = copy.deepcopy(units)
    counted[2]["ledger_events"] = 3
    result = src.useful_attempt_rows(report, counted, clock_id=CLOCK_ID, scope="synthetic")
    assert result["classification"]["job-c/attempt-1"]["useful"] is None
    assert "unit_record_event_count_mismatch" in codes(result, "error")

    doubled = units + [copy.deepcopy(units[0])]
    result = src.useful_attempt_rows(report, doubled, clock_id=CLOCK_ID, scope="synthetic")
    assert result["classification"]["job-a/attempt-1"]["reason"] == "unit_record_duplicated"

    orphan = units + [unit_record(3, "job-z", [started("job-z", 1, 1)])]
    result = src.useful_attempt_rows(report, orphan, clock_id=CLOCK_ID, scope="synthetic")
    assert "unit_record_without_ledger_attempts" in codes(result, "error")


def test_failed_attempt_is_never_useful_even_with_a_complete_record():
    events = [started("job-b", 1, 1410), terminal("job-b", 1, "failed", 1450)]
    _, useful = ledger_and_rows(events)
    assert useful["rows"][0]["useful"] is False
    assert useful["rows"][0]["status"] == "failed"


def test_unknown_ledger_issue_code_is_treated_as_integrity(monkeypatch):
    # A ledger version that emits a code this module does not know: the report is
    # self-consistent (re-derivation uses the same reconcile_attempts), the code impugns it.
    def future_reconcile(schedule, events):
        report = reconcile_attempts(schedule, events)
        report["issues"].append({"code": "brand_new_code", "path": "x", "message": "m"})
        return report

    monkeypatch.setattr(src, "reconcile_attempts", future_reconcile)
    events = scenario1_events()
    report = future_reconcile([job(n) for n in ("job-a", "job-b", "job-c")], events)
    units = [unit_record(i, n, events) for i, n in enumerate(("job-a", "job-b", "job-c"))]
    useful = src.useful_attempt_rows(report, units, clock_id=CLOCK_ID, scope="synthetic")
    assert {v["useful"] for v in useful["classification"].values()} == {None}
    assert useful["status"] == "blocked"


def test_real_non_outcome_ledger_code_is_treated_as_integrity():
    events = scenario1_events()
    events[2], events[3] = (dict(events[2], timestamp_us=1390 * S),
                            dict(events[3], timestamp_us=1395 * S))  # regress below 1400
    report = reconcile_attempts([job(n) for n in ("job-a", "job-b", "job-c")], events)
    assert "timestamp_regression" in {item["code"] for item in report["issues"]}
    units = [unit_record(i, n, events) for i, n in enumerate(("job-a", "job-b", "job-c"))]
    useful = src.useful_attempt_rows(report, units, clock_id=CLOCK_ID, scope="synthetic")
    assert {v["useful"] for v in useful["classification"].values()} == {None}
    assert "ledger_integrity_issue" in codes(useful, "error")


def test_report_that_contradicts_its_own_events_fails_closed():
    events = scenario1_events()
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], events)
    units = [unit_record(i, n, events) for i, n in enumerate(names)]
    # Forged attempts: job-b claims an established completion its raw events never record.
    forged = copy.deepcopy(report)
    attempt = next(a for a in forged["attempts"] if a["attempt_id"] == "job-b/attempt-1")
    done = terminal("job-b", 1, "completed", 1450)
    attempt.update(state="completed", terminal_reason="process_completed",
                   upstream_identity=done["upstream_identity"])
    attempt["events"] = [attempt["events"][0], done]
    forged["issues"] = [i for i in forged["issues"] if i["path"] != "job:job-b"]
    units[1]["steps"][2]["attempts"] = [{"attempt_id": "job-b/attempt-1", "type": "completed",
                                         "reason": "process_completed"}]
    # Forged issue only, and an int timestamp turned into an equal float.
    issue_only = copy.deepcopy(report)
    issue_only["issues"].append({"code": "brand_new_code", "path": "x", "message": "m"})
    as_float = copy.deepcopy(report)
    as_float["counts"]["attempts"] = float(as_float["counts"]["attempts"])
    for bad in (forged, issue_only, as_float):
        with pytest.raises(ValueError):
            src.useful_attempt_rows(bad, units, clock_id=CLOCK_ID, scope="synthetic")


def test_inputs_that_cannot_be_canonically_hashed_fail_closed():
    events = scenario1_events()
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], events)
    units = [unit_record(i, n, events) for i, n in enumerate(names)]
    nan_unit = copy.deepcopy(units)
    nan_unit[0]["units"]["timeout_s"] = float("nan")
    set_unit = copy.deepcopy(units)
    set_unit[1]["producer"]["source_sha256"] = {1, 2}
    inf_report = copy.deepcopy(report)
    inf_report["counts"]["scheduled_jobs"] = float("inf")
    for bad_report, bad_units in ((report, nan_unit), (report, set_unit), (inf_report, units)):
        with pytest.raises(ValueError):
            src.useful_attempt_rows(bad_report, bad_units, clock_id=CLOCK_ID,
                                    scope="synthetic")


def test_useful_rows_reject_bad_inputs():
    events = scenario1_events()
    report = reconcile_attempts([job(n) for n in ("job-a", "job-b", "job-c")], events)
    units = [unit_record(i, n, events) for i, n in enumerate(("job-a", "job-b", "job-c"))]
    bad_report = dict(report)
    bad_report.pop("events")
    wrong_scope = copy.deepcopy(units)
    wrong_scope[0]["scope"] = "actual"
    wrong_kind = copy.deepcopy(units)
    wrong_kind[0]["artifact_kind"] = "s2_unit_run_summary"
    for args, kwargs in [
        ((bad_report, units), {}),
        ((report, wrong_scope), {}),
        ((report, wrong_kind), {}),
        ((report, units), {"clock_id": ""}),
        ((report, units), {"scope": "real"}),
        ((report, "units"), {}),
    ]:
        params = {"clock_id": CLOCK_ID, "scope": "synthetic", **kwargs}
        with pytest.raises(ValueError):
            src.useful_attempt_rows(*args, **params)


def test_artifacts_carry_the_shared_envelope(tmp_path):
    events = scenario1_events()
    _, useful = ledger_and_rows(events)
    part = partition(tmp_path, events, useful["rows"])
    scrape = parse(energy_scrape("1"))
    for artifact in (useful, part, scrape):
        for key in ("artifact_kind", "schema_version", "scope", "producer", "input_hashes",
                    "units", "status", "issues"):
            assert key in artifact
        assert artifact["producer"]["module"] == "gate.s2_telemetry_sources"
        assert len(artifact["producer"]["source_sha256"]) == 64
        assert artifact["status"] in ("passed", "blocked", "unverified")
        json.dumps(artifact, allow_nan=False)


def test_genuine_report_over_non_json_input_is_blocked_not_raised():
    # F6 residual (d): reconcile_attempts stores a non-JSON input value (timestamp_us NaN)
    # as null with an invalid_json_value issue. Re-deriving from the stored (nulled) events
    # cannot reproduce that issue, so the report is accepted as genuine only after the
    # normalisation below, and every flag is None (blocked), never a raise.
    events = scenario1_events()
    events[3] = dict(events[3], timestamp_us=float("nan"))  # job-b's failed terminal
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], events)
    assert "invalid_json_value" in {item["code"] for item in report["issues"]}
    assert report["events"][3]["timestamp_us"] is None
    units = [unit_record(i, n, scenario1_events()) for i, n in enumerate(names)]
    useful = src.useful_attempt_rows(report, units, clock_id=CLOCK_ID, scope="synthetic")
    assert useful["status"] == "blocked"
    assert {"ledger_input_non_json", "ledger_integrity_issue"} <= codes(useful, "error")
    assert {v["useful"] for v in useful["classification"].values()} == {None}
    assert {v["reason"] for v in useful["classification"].values()} == {
        "ledger_integrity_issue"}
    assert useful["counts"] == {"attempts": 3, "useful_true": 0, "useful_false": 0,
                                "useful_unknown": 3, "rejected": 0}
    # job-b's terminal was nulled, so its attempt stays incomplete (unresolved).
    assert [(r["attempt_id"], r["status"], r["end_us"]) for r in useful["rows"]] == [
        ("job-a/attempt-1", "completed", 1400 * S), ("job-b/attempt-1", "unresolved", None),
        ("job-c/attempt-1", "completed", 1700 * S)]
    json.dumps(useful, allow_nan=False)
    alloc = {"allocation_id": "a", "clock_id": CLOCK_ID, "gpu_count": 1, "devices": ["gpu0"],
             "billed_start_us": 1000 * S, "billed_end_us": 2000 * S,
             "segments": [{"phase": "active", "start_us": 1000 * S, "end_us": 2000 * S}]}
    summary = tel.summarize_telemetry([], alloc, useful["rows"], clock_metadata=CLOCK,
                                      max_sample_gap_us=S, scope="synthetic")
    assert summary["useful_work"]["useful_unknown"] == 3
    assert summary["useful_work"]["status"] == "unverified"


def test_genuine_report_over_a_non_list_events_container_is_blocked_not_raised():
    # G4-R5: events passed as a tuple are stored as null (invalid_json_value at "events",
    # plus invalid_events). The genuine report is blocked with no rows, never raised.
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], tuple(scenario1_events()))
    assert report["events"] is None
    assert {"code": "invalid_json_value", "path": "events"} in [
        {"code": i["code"], "path": i["path"]} for i in report["issues"]]
    units = [unit_record(i, n, scenario1_events()) for i, n in enumerate(names)]
    useful = src.useful_attempt_rows(report, units, clock_id=CLOCK_ID, scope="synthetic")
    assert useful["status"] == "blocked"
    assert {"ledger_input_non_json", "ledger_integrity_issue"} <= codes(useful, "error")
    assert useful["rows"] == [] and useful["classification"] == {}
    assert useful["counts"] == {"attempts": 0, "useful_true": 0, "useful_false": 0,
                                "useful_unknown": 0, "rejected": 0}
    json.dumps(useful, allow_nan=False)
    # Null events without that issue are still a malformed report (raised, fail closed).
    forged = copy.deepcopy(reconcile_attempts([job(n) for n in names], scenario1_events()))
    forged["events"] = None
    with pytest.raises(ValueError):
        src.useful_attempt_rows(forged, units, clock_id=CLOCK_ID, scope="synthetic")
    # The issue at another path does not license null events either.
    other = copy.deepcopy(report)
    for item in other["issues"]:
        if item["code"] == "invalid_json_value":
            item["path"] = "events[0].x"
    with pytest.raises(ValueError):
        src.useful_attempt_rows(other, units, clock_id=CLOCK_ID, scope="synthetic")


def test_non_json_report_that_contradicts_its_own_events_still_fails_closed():
    events = scenario1_events()
    events[3] = dict(events[3], timestamp_us=float("nan"))
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], events)
    units = [unit_record(i, n, scenario1_events()) for i, n in enumerate(names)]
    forged = copy.deepcopy(report)
    attempt = next(a for a in forged["attempts"] if a["attempt_id"] == "job-a/attempt-1")
    attempt["terminal_timestamp_us"] = 1399 * S  # disagrees with the stored events
    extra = copy.deepcopy(report)
    extra["issues"].append({"code": "invalid_json_value", "path": "events[0].x",
                            "message": "forged"})
    extra["issues"].append({"code": "brand_new_code", "path": "x", "message": "m"})
    for bad in (forged, extra):
        with pytest.raises(ValueError):
            src.useful_attempt_rows(bad, units, clock_id=CLOCK_ID, scope="synthetic")


def test_forged_invalid_json_issue_on_a_clean_report_only_blocks():
    # Adding an invalid_json_value issue to an otherwise genuine report can only blank every
    # flag: forging it never yields a useful episode.
    events = scenario1_events()
    names = ("job-a", "job-b", "job-c")
    report = reconcile_attempts([job(n) for n in names], events)
    units = [unit_record(i, n, events) for i, n in enumerate(names)]
    forged = copy.deepcopy(report)
    forged["admitted"] = False
    forged["issues"].insert(0, {"code": "invalid_json_value", "path": "events[0].x",
                                "message": "Non-JSON or cyclic value retained as null."})
    useful = src.useful_attempt_rows(forged, units, clock_id=CLOCK_ID, scope="synthetic")
    assert useful["status"] == "blocked"
    assert {v["useful"] for v in useful["classification"].values()} == {None}


def test_useful_rule_is_labelled_an_agent_technical_default_in_both_modules():
    # F6 residual (c): the producer's useful rule is an agent technical default (D-018);
    # the consumer takes the flag as given and says so.
    label = "agent technical default (D-018)"
    assert label in src.__doc__ and label in tel.__doc__
    _, useful = ledger_and_rows(scenario1_events())
    assert useful["definition"].startswith(label + ": ")
    alloc = {"allocation_id": "a", "clock_id": CLOCK_ID, "gpu_count": 1, "devices": ["gpu0"],
             "billed_start_us": 1000 * S, "billed_end_us": 2000 * S,
             "segments": [{"phase": "active", "start_us": 1000 * S, "end_us": 2000 * S}]}
    summary = tel.summarize_telemetry([], alloc, useful["rows"], clock_metadata=CLOCK,
                                      max_sample_gap_us=S, scope="synthetic")
    assert label in summary["useful_work"]["definition"]
