"""Fixed-interval nvidia-smi sampler emitting the ``gate.s2_telemetry`` sample schema.

Tickets: build-spec B1 (lane R1 P6 "in-run telemetry sampler"; readiness CPU-07 "in-run
sampler" feeding A6 ``gate.s2_telemetry.summarize_telemetry``). Agent-authored under
D-018/D-024; serves S2.7 [C] (G7).

Each tick runs one query through an injected runner::

    nvidia-smi --query-gpu=index,uuid,power.draw,memory.used --format=csv

and emits, for every caller-declared device and each metric in ``METRICS``
(``nvidia_smi:power.draw`` in W, ``nvidia_smi:memory.used`` in MiB; both in
``gate.s2_telemetry.PINNED_METRICS``), exactly one sample
``{metric, device, clock_id, t_us, status, value, raw_ref}``:

- ``t_us`` is the injected monotonic clock read (integer microseconds) immediately before
  the query is issued; ``clock_id`` names that clock (required input).
- Output format: the header line must be exactly ``HEADER``
  (``index, uuid, power.draw [W], memory.used [MiB]``) and values carry their unit
  (``18.15 W``, ``0 MiB``). Every token is anchored to a recorded host log
  (``evidence/s2-vast-host-20261101-v1``): ``index``/``uuid`` in probe-v2/gpu_identity.log,
  ``power.draw [W]`` and ``18.15 W`` in probe-v2/gpu_telemetry.log, ``memory.used [MiB]``
  and ``0 MiB`` in monitor-v1/nvml-after.log, all from ``--format=csv``. A missing or
  different header, a missing or wrong unit, or a unitless value is ``scrape_failed``.
- ``status`` is ``ok`` (finite numeric value) or, with ``value: None``: ``not_supported``
  (``[N/A]``/``[Not Supported]``), ``no_permission`` (``[Insufficient Permissions]``),
  ``no_data`` (the declared device is absent from a successful query) or
  ``scrape_failed`` (timeout, nonzero exit, runner error, malformed or duplicated output,
  any other bracketed or unparseable value). A missing value is never zero.
- ``raw_ref`` is ``<raw file>#tick=<k>`` (plus ``:row=<i>`` when a row was parsed); the raw
  file keeps each tick's exit status and stdout so every sample can be re-derived.

Devices: ``device_key`` (``uuid`` or ``index``, required, no default) chooses which query
column identifies a device, and ``devices`` (required) must equal the allocation's
``devices`` in ``summarize_telemetry``. Rows for undeclared devices are counted, not
emitted.

Schedule: tick ``k`` is due at ``start + k * interval``. A tick whose query overruns the
next due time skips the missed ticks (counted in ``missed_ticks``), never back-fills them.
Sampling ends when the next due tick lies at or after ``start + duration`` or when the
injected ``should_stop`` returns true. Interval, duration and per-query timeout are
required inputs (no defaults); the timeout must be shorter than the interval.

Summary ``status``: ``not_run`` without ticks; ``passed`` only when every declared
device x metric has at least one ``ok`` sample; otherwise ``unverified`` with a blocking
``no_ok_samples`` issue naming the pairs (absence is never a pass).

Verification level: ``parse_query_output`` and ``sample`` L1 with a fake runner and fake
clock (known answers; samples validated by feeding them to ``summarize_telemetry``);
the per-field header names and value formats are L2 against the recorded host logs
above, while the combined four-field output of this exact query was never recorded:
the summary says ``query_format: qualify_on_host``. Actual cadence, driver behaviour and
value semantics on the rented host are L3 (``qualify_on_host``).

This module never chooses devices or intervals, never interpolates or fills a gap, never
reads credentials, never contacts a network and never runs anything except the fixed
nvidia-smi query argv through the runner it is given.
"""

import argparse
import csv
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_telemetry import PINNED_METRICS, SAMPLE_STATUSES

ARTIFACT_KIND = "s2_gpu_samples"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
DEVICE_KEYS = ("uuid", "index")
QUERY_FIELDS = ("index", "uuid", "power.draw", "memory.used")
QUERY_ARGV = ("nvidia-smi", "--query-gpu=" + ",".join(QUERY_FIELDS), "--format=csv")
FIELD_UNITS = {"power.draw": "W", "memory.used": "MiB"}
HEADER = ", ".join(f"{field} [{FIELD_UNITS[field]}]" if field in FIELD_UNITS else field
                   for field in QUERY_FIELDS)
METRICS = {"power.draw": "nvidia_smi:power.draw", "memory.used": "nvidia_smi:memory.used"}
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL")
RAW_NAME = "nvidia-smi-raw.jsonl"
SAMPLES_NAME = "samples.jsonl"
SUMMARY_NAME = "sampler-summary.json"
_BRACKETED = {"[N/A]": "not_supported", "[Not Supported]": "not_supported",
              "[Insufficient Permissions]": "no_permission"}
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
assert all(metric in PINNED_METRICS for metric in METRICS.values())
assert {"ok", "not_supported", "no_permission", "no_data", "scrape_failed"} <= set(
    SAMPLE_STATUSES)


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _value(text, field):
    """(status, value) for one ``<number> <unit>`` CSV cell; never coerces missing to zero."""
    if text in _BRACKETED:
        return _BRACKETED[text], None
    number, separator, unit = text.rpartition(" ")
    if not separator or unit != FIELD_UNITS[field] or _NUMBER.fullmatch(number) is None:
        return "scrape_failed", None
    value = float(number)
    return ("ok", value) if math.isfinite(value) else ("scrape_failed", None)


def parse_query_output(stdout, *, device_key):
    """Parse one successful query's stdout into ``{device: (row_index, {field: cell})}``.

    Raises ``ValueError`` when any line is malformed or a device appears twice: the whole
    scrape is then untrustworthy and every declared device reads ``scrape_failed``.
    """
    if device_key not in DEVICE_KEYS:
        raise ValueError(f"device_key must be one of {DEVICE_KEYS}")
    rows = {}
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines or lines[0].strip() != HEADER:
        raise ValueError(f"first line is not the expected header {HEADER!r}")
    lines = lines[1:]
    for number, row in enumerate(csv.reader(io.StringIO("\n".join(lines)),
                                            skipinitialspace=True)):
        if len(row) != len(QUERY_FIELDS) or not all(cell.strip() for cell in row):
            raise ValueError(f"row {number} does not have {len(QUERY_FIELDS)} fields")
        cells = dict(zip(QUERY_FIELDS, (cell.strip() for cell in row), strict=True))
        if re.fullmatch(r"\d+", cells["index"]) is None:
            raise ValueError(f"row {number} has a non-integer index")
        device = cells[device_key]
        if device in rows:
            raise ValueError(f"device {device!r} appears twice")
        rows[device] = (number, cells)
    return rows


def tick_samples(result, *, tick, t_us, devices, device_key, clock_id, raw_name=RAW_NAME):
    """Samples for one tick from a runner result ``{returncode, stdout, timed_out}`` (L1)."""
    base_ref = f"{raw_name}#tick={tick}"
    rows, failure = {}, None
    if result is None or result.get("timed_out") or result.get("returncode") != 0:
        failure = "query_failed"
    else:
        try:
            rows = parse_query_output(result.get("stdout") or "", device_key=device_key)
        except ValueError:
            failure = "malformed_output"
    samples = []
    for device in devices:
        for field, metric in METRICS.items():
            ref = base_ref
            if failure:
                status, value = "scrape_failed", None
            elif device not in rows:
                status, value = "no_data", None
            else:
                number, cells = rows[device]
                status, value = _value(cells[field], field)
                ref = f"{base_ref}:row={number}"
            samples.append({"metric": metric, "device": device, "clock_id": clock_id,
                            "t_us": t_us, "status": status, "value": value, "raw_ref": ref})
    unallocated = 0 if failure else len(set(rows) - set(devices))
    return samples, unallocated


def _check_inputs(devices, device_key, clock_id, interval_s, duration_s, timeout_s, scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; no default is chosen")
    if device_key not in DEVICE_KEYS:
        raise ValueError(f"device_key must be one of {DEVICE_KEYS}; no default is chosen")
    if (type(devices) is not list or not devices or len(set(devices)) != len(devices)
            or not all(type(d) is str and d.strip() and d == d.strip() for d in devices)):
        raise ValueError("devices must be a nonempty list of distinct device IDs")
    if type(clock_id) is not str or not clock_id.strip():
        raise ValueError("clock_id must name the monotonic clock")
    for name, value in (("interval_s", interval_s), ("duration_s", duration_s),
                        ("timeout_s", timeout_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if timeout_s >= interval_s:
        raise ValueError("timeout_s must be shorter than interval_s")


def _append(stream, record):
    stream.write(canonical_json_bytes(record) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def sample(output_dir, *, runner, clock, devices, device_key, clock_id, interval_s,
           duration_s, timeout_s, scope, utc_anchor=None, should_stop=None):
    """Run the fixed-interval loop; write samples, raw output and a summary (L1 with fakes).

    ``runner(argv, timeout_s)`` returns ``{returncode, stdout, timed_out}``;
    ``clock.monotonic_us()`` returns integer microseconds; ``clock.sleep(seconds)``.
    ``output_dir`` must not exist. Returns the summary artifact.
    """
    _check_inputs(devices, device_key, clock_id, interval_s, duration_s, timeout_s, scope)
    if utc_anchor is not None and (type(utc_anchor) is not str or not utc_anchor.strip()):
        raise ValueError("utc_anchor must be text or None")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    interval_us = round(interval_s * 1_000_000)
    duration_us = round(duration_s * 1_000_000)
    statuses = {status: 0 for status in SAMPLE_STATUSES}
    ticks = missed = unallocated_rows = runner_errors = 0
    start = clock.monotonic_us()
    tick = 0
    with open(output / SAMPLES_NAME, "xb") as samples_file, \
            open(output / RAW_NAME, "xb") as raw_file:
        while tick * interval_us < duration_us:
            if should_stop is not None and should_stop():
                break
            due = start + tick * interval_us
            now = clock.monotonic_us()
            if now < due:
                clock.sleep((due - now) / 1_000_000)
            t_us = clock.monotonic_us()
            try:
                result = runner(list(QUERY_ARGV), timeout_s)
                result = {"returncode": result.get("returncode"),
                          "stdout": result.get("stdout") or "",
                          "timed_out": bool(result.get("timed_out"))}
            except Exception as exc:  # noqa: BLE001 - a runner failure is a scrape failure
                runner_errors += 1
                result = None
                _append(raw_file, {"tick": tick, "t_us": t_us, "runner_error":
                                   type(exc).__name__})
            if result is not None:
                _append(raw_file, {"tick": tick, "t_us": t_us, **result})
            rows, extra = tick_samples(result, tick=tick, t_us=t_us, devices=devices,
                                       device_key=device_key, clock_id=clock_id)
            unallocated_rows += extra
            for row in rows:
                statuses[row["status"]] += 1
                _append(samples_file, row)
            ticks += 1
            elapsed = clock.monotonic_us() - start
            following = max(tick + 1, -(-elapsed // interval_us))
            missed += following - tick - 1
            tick = following
    samples = load_samples(output / SAMPLES_NAME)
    summary = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "scripts.s2_gpu_sampler", "source_sha256": _source_sha256()},
        "input_hashes": {"samples_jsonl": hashlib.sha256(
            (output / SAMPLES_NAME).read_bytes()).hexdigest(),
            "raw_jsonl": hashlib.sha256((output / RAW_NAME).read_bytes()).hexdigest()},
        "clock_metadata": {"clock_id": clock_id, "clock_kind": "monotonic",
                           "clock_scope": "host", "utc_anchor": utc_anchor},
        "query_argv": list(QUERY_ARGV), "device_key": device_key, "devices": list(devices),
        "metrics": {metric: PINNED_METRICS[metric][0] for metric in METRICS.values()},
        "units": {"t_us": "microseconds", "interval_us": "microseconds"},
        "interval_us": interval_us, "duration_us": duration_us, "ticks": ticks,
        "missed_ticks": missed, "runner_errors": runner_errors,
        "unallocated_rows": unallocated_rows, "status_counts": statuses,
        "sample_count": len(samples), "samples_sha256": identity_hash(samples),
        "query_format": "qualify_on_host", "status": "not_run", "issues": [],
    }
    ok_pairs = {(row["device"], row["metric"]) for row in samples if row["status"] == "ok"}
    never_ok = sorted([device, metric] for device in devices for metric in METRICS.values()
                      if (device, metric) not in ok_pairs)
    if ticks:
        summary["status"] = "unverified" if never_ok else "passed"
    if missed:
        summary["issues"].append({"code": "missed_ticks", "path": "ticks", "severity": "warning",
                                  "evidence_refs": [RAW_NAME],
                                  "message": f"{missed} due ticks skipped after overruns"})
    if ticks and never_ok:
        summary["issues"].append({
            "code": "no_ok_samples", "path": "samples", "severity": "blocking",
            "evidence_refs": [SAMPLES_NAME, RAW_NAME],
            "message": f"No ok sample for device x metric pairs {never_ok}"})
    with open(output / SUMMARY_NAME, "xb") as stream:
        stream.write(canonical_json_bytes(summary) + b"\n")
    return summary


def load_samples(path):
    """Read a samples JSONL file back into the list ``summarize_telemetry`` consumes."""
    return [read_json_bytes(line) for line in Path(path).read_bytes().splitlines() if line]


def subprocess_runner(argv, timeout_s):
    """Real runner: no shell, closed stdin, allowlisted environment (L3 on the host)."""
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    try:
        completed = subprocess.run(list(argv), stdin=subprocess.DEVNULL, capture_output=True,
                                   timeout=timeout_s, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "timed_out": True}
    return {"returncode": completed.returncode,
            "stdout": completed.stdout.decode("utf-8", errors="replace"), "timed_out": False}


class SystemClock:
    @staticmethod
    def monotonic_us():
        return time.monotonic_ns() // 1000

    @staticmethod
    def sleep(seconds):
        time.sleep(max(0.0, seconds))


def main(argv=None, *, runner=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", help="new directory for samples and raw output")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--device-key", required=True, choices=DEVICE_KEYS)
    parser.add_argument("--device", action="append", required=True, dest="devices")
    parser.add_argument("--clock-id", required=True)
    parser.add_argument("--interval-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--utc-anchor")
    args = parser.parse_args(argv)
    summary = sample(args.output_dir, runner=runner or subprocess_runner,
                     clock=clock or SystemClock(), devices=args.devices,
                     device_key=args.device_key, clock_id=args.clock_id,
                     interval_s=args.interval_s, duration_s=args.duration_s,
                     timeout_s=args.timeout_s, scope=args.scope, utc_anchor=args.utc_anchor)
    print(f"{summary['status']}: {summary['ticks']} ticks, {summary['sample_count']} samples")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
