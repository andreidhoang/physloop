"""Fixed-interval in-run scraper of the loopback dcgm-exporter ``/metrics`` endpoint.

Tickets: completeness-audit F6 residual (a) (an in-run guest producer for
``gate.s2_telemetry_sources.parse_dcgm_scrapes``; readiness CPU-07 / S2.7 [C], G7). Modeled on
``scripts/s2_gpu_sampler.py``. Agent-authored evaluator engineering under D-018/D-024.

Endpoint. ``scripts/vast_exporter_control.sh`` starts the pinned exporter image with
``-a 127.0.0.1:9400 -c 1000 -f /etc/dcgm-exporter/default-counters.csv`` (:128-131) and scrapes
it with ``curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9400/metrics``
(:135, :154). Each tick here runs that same request through an injected runner::

    curl --disable --fail --silent --show-error --max-time <timeout_s> \
        http://127.0.0.1:9400/metrics

``--disable`` (first argument, so curl reads no ``.curlrc``) is the one labelled deviation from
the control script; ``--max-time`` is the required ``timeout_s`` input instead of a fixed 2.

Output (``output_dir`` must not exist):

- ``dcgm-scrapes.jsonl``: one record per tick, exactly ``{t_us, text, ref}``, the input
  ``parse_dcgm_scrapes`` takes. ``t_us`` is the injected monotonic clock read (integer
  microseconds) immediately before the request. ``text`` is the response body decoded as strict
  UTF-8 and stored unstripped (the parser needs the trailing newline and hashes the exact text),
  or ``None`` when no trustworthy body exists: ``timed_out``, ``nonzero_exit`` (curl ``--fail``
  turns an HTTP error into an exit code), ``runner_error``, ``invalid_runner_result`` or
  ``undecodable_body``. An empty body from a successful request is kept as ``""`` and the
  parser reports it (``scrape_empty``). ``ref`` is ``dcgm-raw.jsonl#tick=<k>``.
- ``dcgm-raw.jsonl``: per tick ``{tick, t_us, returncode, timed_out, body_bytes, body_sha256,
  failure}`` (no stderr, no body copy), so every record can be re-checked against its bytes.
- ``dcgm-scraper-summary.json``: the artifact envelope with tick, miss and failure counts.

Schedule: tick ``k`` is due at ``start + k * interval``. A request that overruns the next due
time skips the missed ticks (``missed_ticks``), never back-fills them. Sampling ends when the
next due tick lies at or after ``start + duration`` or the injected ``should_stop`` returns
true. A clock read that does not increase stops the loop before that record is written
(``clock_not_increasing``, status ``blocked``), so the file always has strictly increasing
times. Interval, duration and timeout are required inputs (no defaults); the loop runs in whole
microseconds, so interval and duration must each round to at least 1 us and the timeout must be
shorter than the interval after rounding (checked before the output directory is created).

Summary ``status``: ``not_run`` without ticks; ``passed`` when every tick captured a body
(capture only: parsing, HELP/TYPE pins and device mapping are ``parse_dcgm_scrapes``'s job);
``unverified`` with a ``scrape_failures`` warning when some ticks failed, or with a blocking
``no_scrape_captured`` issue when none succeeded; ``blocked`` on a non-increasing clock.

Verification level: ``scrape_argv``, ``tick_record`` and ``scrape`` are L1 with a fake runner
and fake clock (known answers; records fed to ``parse_dcgm_scrapes`` with the recorded bodies).
The endpoint and argv are L2 against the control script lines above. The exporter's behaviour
under a real run (cadence, XID delivery, endpoint availability while workloads run) is L3
(``qualify_on_host``). Integration gap (recorded, not closed here): the control script removes
its exporter container on exit (cleanup trap, :114-119), and no plan stage yet starts a
run-long exporter or invokes this scraper, so during a run the endpoint has no producer until
that stage and its wiring are added.

This module never chooses an interval or a device mapping, never parses or interprets metric
values, never fills a gap or invents a body, never reads credentials, never contacts any
address other than the fixed loopback endpoint, and never runs anything except that fixed curl
argv through the runner it is given.
"""

import argparse
import hashlib
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import canonical_json_bytes, read_json_bytes

ARTIFACT_KIND = "s2_dcgm_scrape_capture"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
ENDPOINT = "http://127.0.0.1:9400/metrics"  # scripts/vast_exporter_control.sh:130,135,154
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL")
SCRAPES_NAME = "dcgm-scrapes.jsonl"
RAW_NAME = "dcgm-raw.jsonl"
SUMMARY_NAME = "dcgm-scraper-summary.json"
PARSER = "gate.s2_telemetry_sources.parse_dcgm_scrapes"
FAILURES = ("invalid_runner_result", "nonzero_exit", "runner_error", "timed_out",
            "undecodable_body")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _issue(code, path, severity, message, refs):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": list(refs),
            "message": message}


def scrape_argv(timeout_s):
    """The fixed curl argv for one scrape (L2 against the exporter control script)."""
    return ["curl", "--disable", "--fail", "--silent", "--show-error", "--max-time",
            f"{timeout_s:g}", ENDPOINT]


def _classify(result):
    """(failure, body bytes or None) for one runner result; never raises."""
    if type(result) is not dict or set(result) != {"returncode", "stdout", "timed_out"}:
        return "invalid_runner_result", None
    returncode, stdout, timed_out = result["returncode"], result["stdout"], result["timed_out"]
    if type(timed_out) is not bool or (returncode is not None and type(returncode) is not int):
        return "invalid_runner_result", None
    if type(stdout) not in (bytes, bytearray):
        return "invalid_runner_result", None
    if timed_out:
        return "timed_out", bytes(stdout)
    if returncode != 0:
        return "nonzero_exit", bytes(stdout)
    return None, bytes(stdout)


def tick_record(result, *, tick, t_us, raw_name=RAW_NAME):
    """``({t_us, text, ref}, raw line)`` for one tick's runner result (L1).

    ``result`` is ``{returncode, stdout (bytes), timed_out}`` or an exception instance
    raised by the runner. ``text`` is the strict-UTF-8 body or ``None``.
    """
    if isinstance(result, BaseException):
        failure, body = "runner_error", None
    else:
        failure, body = _classify(result)
    text = None
    if failure is None:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            failure = "undecodable_body"
    plain = type(result) is dict and failure != "invalid_runner_result"
    raw = {"tick": tick, "t_us": t_us,
           "returncode": result["returncode"] if plain else None,
           "timed_out": result["timed_out"] if plain else None,
           "body_bytes": None if body is None else len(body),
           "body_sha256": None if body is None else hashlib.sha256(body).hexdigest(),
           "failure": failure}
    return {"t_us": t_us, "text": text, "ref": f"{raw_name}#tick={tick}"}, raw


def _check_inputs(clock_id, interval_s, duration_s, timeout_s, scope, utc_anchor):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; no default is chosen")
    if type(clock_id) is not str or not clock_id.strip():
        raise ValueError("clock_id must name the monotonic clock")
    for name, value in (("interval_s", interval_s), ("duration_s", duration_s),
                        ("timeout_s", timeout_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if timeout_s >= interval_s:
        raise ValueError("timeout_s must be shorter than interval_s")
    if utc_anchor is not None and (type(utc_anchor) is not str or not utc_anchor.strip()):
        raise ValueError("utc_anchor must be text or None")
    interval_us, duration_us, timeout_us = (_us(v) for v in (interval_s, duration_s, timeout_s))
    if interval_us < 1 or duration_us < 1:
        raise ValueError("interval_s and duration_s must each be at least 1 microsecond")
    if timeout_us >= interval_us:
        raise ValueError("timeout_s must be shorter than interval_s in whole microseconds")
    return interval_us, duration_us


def _us(seconds):
    return round(seconds * 1_000_000)


def _append(stream, record):
    stream.write(canonical_json_bytes(record) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def scrape(output_dir, *, runner, clock, clock_id, interval_s, duration_s, timeout_s, scope,
           utc_anchor=None, should_stop=None):
    """Run the fixed-interval loop; write records, raw lines and a summary (L1 with fakes).

    ``runner(argv, timeout_s)`` returns ``{returncode, stdout, timed_out}`` with ``stdout``
    bytes; ``clock.monotonic_us()`` returns integer microseconds; ``clock.sleep(seconds)``.
    Returns the summary artifact.
    """
    interval_us, duration_us = _check_inputs(clock_id, interval_s, duration_s, timeout_s, scope,
                                             utc_anchor)
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    argv = scrape_argv(timeout_s)
    failures = {}
    ticks = missed = captured = 0
    previous = None
    clock_stuck = False
    start = clock.monotonic_us()
    tick = 0
    with open(output / SCRAPES_NAME, "xb") as scrapes_file, \
            open(output / RAW_NAME, "xb") as raw_file:
        while tick * interval_us < duration_us:
            if should_stop is not None and should_stop():
                break
            due = start + tick * interval_us
            now = clock.monotonic_us()
            if now < due:
                clock.sleep((due - now) / 1_000_000)
            t_us = clock.monotonic_us()
            if type(t_us) is not int or (previous is not None and t_us <= previous):
                clock_stuck = True
                break
            previous = t_us
            try:
                result = runner(list(argv), timeout_s)
            except Exception as exc:  # noqa: BLE001 - a runner failure is a failed scrape
                result = exc
            record, raw = tick_record(result, tick=tick, t_us=t_us)
            _append(raw_file, raw)
            _append(scrapes_file, record)
            if raw["failure"] is None:
                captured += 1
            else:
                failures[raw["failure"]] = failures.get(raw["failure"], 0) + 1
            ticks += 1
            elapsed = clock.monotonic_us() - start
            following = max(tick + 1, -(-elapsed // interval_us))
            missed += following - tick - 1
            tick = following
    issues = []
    if missed:
        issues.append(_issue("missed_ticks", "ticks", "warning",
                             f"{missed} due ticks skipped after overruns", [RAW_NAME]))
    if clock_stuck:
        issues.append(_issue("clock_not_increasing", "ticks", "blocking", "The monotonic clock "
                             "did not increase; sampling stopped before writing that record",
                             [SCRAPES_NAME]))
    if ticks and failures and captured:
        issues.append(_issue("scrape_failures", "ticks", "warning",
                             f"{ticks - captured} of {ticks} scrapes captured no body",
                             [RAW_NAME]))
    if ticks and not captured:
        issues.append(_issue("no_scrape_captured", "ticks", "blocking",
                             "No scrape captured a body", [SCRAPES_NAME, RAW_NAME]))
    if clock_stuck:
        status = "blocked"
    elif not ticks:
        status = "not_run"
    else:
        status = "passed" if not failures else "unverified"
    summary = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "scripts.s2_dcgm_scraper", "source_sha256": _source_sha256()},
        "input_hashes": {
            "scrapes_jsonl": hashlib.sha256((output / SCRAPES_NAME).read_bytes()).hexdigest(),
            "raw_jsonl": hashlib.sha256((output / RAW_NAME).read_bytes()).hexdigest()},
        "clock_metadata": {"clock_id": clock_id, "clock_kind": "monotonic",
                           "clock_scope": "host", "utc_anchor": utc_anchor},
        "endpoint": ENDPOINT, "argv": argv, "parse_with": PARSER,
        "units": {"t_us": "microseconds", "interval_us": "microseconds",
                  "duration_us": "microseconds", "body_bytes": "bytes"},
        "interval_us": interval_us, "duration_us": duration_us, "ticks": ticks,
        "missed_ticks": missed, "captured": captured,
        "failures_by_reason": dict(sorted(failures.items())),
        "status": status,
        "status_meaning": "capture only; parse with " + PARSER + "; never host qualification",
        "host_qualification": "qualify_on_host",
        "issues": issues,
    }
    with open(output / SUMMARY_NAME, "xb") as stream:
        stream.write(canonical_json_bytes(summary) + b"\n")
    return summary


def load_scrapes(path):
    """Read a scrapes JSONL file back into the list ``parse_dcgm_scrapes`` consumes."""
    return [read_json_bytes(line) for line in Path(path).read_bytes().splitlines() if line]


def subprocess_runner(argv, timeout_s):
    """Real runner: no shell, closed stdin, allowlisted environment (L3 on the host)."""
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    try:
        completed = subprocess.run(list(argv), stdin=subprocess.DEVNULL, capture_output=True,
                                   timeout=timeout_s, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": b"", "timed_out": True}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "timed_out": False}


class SystemClock:
    @staticmethod
    def monotonic_us():
        return time.monotonic_ns() // 1000

    @staticmethod
    def sleep(seconds):
        time.sleep(max(0.0, seconds))


def main(argv=None, *, runner=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", help="new directory for scrape records and raw lines")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--clock-id", required=True)
    parser.add_argument("--interval-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--utc-anchor")
    args = parser.parse_args(argv)
    summary = scrape(args.output_dir, runner=runner or subprocess_runner,
                     clock=clock or SystemClock(), clock_id=args.clock_id,
                     interval_s=args.interval_s, duration_s=args.duration_s,
                     timeout_s=args.timeout_s, scope=args.scope, utc_anchor=args.utc_anchor)
    print(f"{summary['status']}: {summary['ticks']} ticks, {summary['captured']} captured")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
