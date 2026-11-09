"""S2 telemetry and useful-work accounting summary (readiness CPU-07; build-spec A6).

Serves S2.7 [C] (G7). Agent-authored evaluator engineering under D-018/D-024.

``summarize_telemetry`` turns already-collected samples (DCGM exporter or
nvidia-smi readings, each with an explicit status), one billed allocation with
its phase partition, and the attempt list into a deterministic summary: a
nonoverlapping allocation-time partition that must sum exactly to billed time,
GPU-hours, useful completed episodes per paid wall-hour and per GPU-hour,
per-series coverage and availability, counter increases with reset events, two
separately labelled energy estimates (power integration and energy-counter
delta), sampled memory high-water marks and XID readings. It never fills a gap,
never reads a missing value as zero and never turns DCGM ``NO_DATA`` into zero
XIDs.

Verification level per function (build-spec section 1):

- ``summarize_telemetry`` - L1 over hand-built synthetic samples with literal
  known answers. Its metric names and units are L2 (pinned to the HELP lines of
  the dcgm-exporter scrape recorded in
  ``evidence/s2-vast-host-20261101-v1/exporter-v1/*.prom`` and the nvidia-smi
  CSV headers recorded in ``evidence/s2-vast-host-20261101-v1/probe-v2/
  gpu_telemetry.log`` (``power.draw [W]``) and ``.../monitor-v1/nvml-after.log``
  (``memory.used [MiB]``); ``DCGM_FI_DEV_XID_ERRORS``, absent from every recorded
  scrape, is pinned to the vendored dcgm-exporter 4.8.4 default counters CSV, see
  ``gate.s2_telemetry_sources``). Actual monitor delivery, sampling cadence and XID event
  delivery are L3 (``qualify_on_host``); the November 1 host delivered no XID
  readings (``DCGM_FI_DEV_XID_ERRORS: []``), so fault observability stays
  ``unverified``.

Interface reconciliation (recorded, not silently chosen): build-spec A6 names
``summarize_telemetry(samples, allocation_events, rollouts)``; readiness CPU-07
names ``summarize_telemetry(samples, allocation_intervals, attempts,
clock_metadata, stabilization_rule=None)``. The positional form follows the
build-spec (the second argument is one allocation carrying its phase segments);
``clock_metadata``, ``stabilization_rule``, ``max_sample_gap_us`` and ``scope``
are keywords.

Input contracts (exact keys, fail closed; times are integer microseconds on one
monotonic clock named by ``clock_id``; UTC is only a correlation anchor and is
never used in arithmetic):

- ``clock_metadata``: ``clock_id``, ``clock_kind`` (``monotonic``),
  ``clock_scope`` (``host`` | ``process``), ``utc_anchor`` (text or ``None``).
- allocation: ``allocation_id``, ``clock_id``, ``gpu_count`` (int >= 1),
  ``devices`` (distinct IDs, one per GPU), ``billed_start_us``, ``billed_end_us``
  and ``segments`` (``phase`` in ``PHASES``, ``start_us`` < ``end_us``).
- sample: ``metric``, ``device``, ``clock_id``, ``t_us``, ``status`` (one of
  ``SAMPLE_STATUSES``), ``value`` (finite number when ``ok``, else ``None``),
  ``raw_ref`` (text reference to the raw scrape or log line).
- rollout attempt: ``attempt_id``, ``job_id``, ``clock_id``, ``start_us``,
  ``end_us`` (``None`` only when ``unresolved``), ``status`` (``completed`` |
  ``failed`` | ``cancelled`` | ``unresolved``) and ``useful`` (``True`` only for
  a completed attempt the ledger/adapter admitted as a useful episode, ``False``,
  or ``None`` when not yet classified). Exit codes are not accepted: success is
  never inferred from a process exit status.

Agent technical defaults (D-018): ``PHASES`` extends CPU-07's startup,
acquisition, warm-up, active, retry, idle and tail with ``export`` (runbook step
8 bills export); trapezoid integration between consecutive ``ok`` power samples
no further apart than ``max_sample_gap_us`` (a required caller input, not
defaulted here); allocation edges before the first and after the last sample are
uncovered rather than extrapolated; counter resets follow the Prometheus
convention (a decrease is a reset; the post-reset value is counted as increase)
and make the increase a lower bound; a reset hidden inside a sampling gap is
undetectable from samples alone, so counter-delta energy across any gap longer
than ``max_sample_gap_us`` is ``unverified`` (``gap_beyond_coverage_limit``) with
its lower bound kept, and so is the per-series counter ``increase``.

Accounting rules: ``input_hashes`` for samples and rollouts cover every supplied
row (order-invariant; a non-JSON row contributes a positional marker), and the
``accepted_*`` hashes cover the rows actually summarized. Dropped samples are
counted by reason; a pinned series whose samples were all dropped outside billed
time reads ``dropped``, not ``absent``; dropping every supplied sample is the
blocking issue ``all_samples_dropped``. Any rejected attempt makes
``useful_work`` ``blocked`` and its rates lower bounds.

Owner inputs (never chosen here): the stabilization rule for stabilized
throughput (``None`` -> ``blocked: stabilization_rule_unregistered``; any rule ->
``unsupported_stabilization_rule`` until a reviewed implementation exists) and
the scientific definition of a useful episode.

Useful flags: this module consumes each attempt's ``useful`` flag and never
infers it. Its producer, ``gate.s2_telemetry_sources.useful_attempt_rows``,
applies an agent technical default (D-018): ``useful=True`` only for an
established completion (``gate.s2_executor.terminal_established``) that is its
job's only attempt with one consistent unit record; failed or cancelled is
``False``; anything else is ``None``. That rule is useful-work accounting, not
the scientific definition, which stays Huy's.

This module never reads files, never runs subprocesses (no nvidia-smi, DCGM or
provider calls), never claims device latency without synchronized device timing
(``device_latency`` is ``not_applicable``) and never imports protected inference.
"""

import hashlib
import itertools
import math
from fractions import Fraction
from pathlib import Path

from gate.s2_identity import identity_hash

ARTIFACT_KIND = "s2_telemetry_summary"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
PHASES = ("startup", "acquisition", "warm_up", "active", "retry", "idle", "export", "tail")
SAMPLE_STATUSES = (
    "ok",
    "no_data",
    "not_supported",
    "not_watched",
    "no_permission",
    "scrape_failed",
)
ROLLOUT_STATUSES = ("completed", "failed", "cancelled", "unresolved")
US_PER_SECOND = 1_000_000
US_PER_HOUR = 3_600_000_000

# metric -> (unit, kind, role). Units from the recorded HELP lines / CSV header.
PINNED_METRICS = {
    "DCGM_FI_DEV_POWER_USAGE": ("W", "gauge", "power"),
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": ("mJ", "counter", "energy"),
    "DCGM_FI_DEV_FB_USED": ("MiB", "gauge", "memory"),
    "DCGM_FI_DEV_GPU_UTIL": ("%", "gauge", None),
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": ("count", "counter", None),
    "DCGM_FI_DEV_XID_ERRORS": ("xid_code", "event", "xid"),
    "nvidia_smi:power.draw": ("W", "gauge", "power"),
    "nvidia_smi:memory.used": ("MiB", "gauge", "memory"),
}

_CLOCK_KEYS = {"clock_id", "clock_kind", "clock_scope", "utc_anchor"}
_ALLOCATION_KEYS = {
    "allocation_id",
    "clock_id",
    "gpu_count",
    "devices",
    "billed_start_us",
    "billed_end_us",
    "segments",
}
_SAMPLE_KEYS = {"metric", "device", "clock_id", "t_us", "status", "value", "raw_ref"}
_ROLLOUT_KEYS = {"attempt_id", "job_id", "clock_id", "start_us", "end_us", "status", "useful"}


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _issue(code, path, message, severity="error"):
    return {
        "code": code,
        "path": path,
        "severity": severity,
        "evidence_refs": [],
        "message": message,
    }


def _text(value):
    return type(value) is str and bool(value.strip()) and "\x00" not in value


def _int(value):
    return type(value) is int


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _quantity(value=None, reason=None, status=None):
    if value is None:
        return {"status": status or "not_estimable", "value": None, "exact": None, "reason": reason}
    return {"status": "passed", "value": float(value), "exact": str(value), "reason": None}


def _check_clock(clock_metadata):
    if type(clock_metadata) is not dict or set(clock_metadata) != _CLOCK_KEYS:
        raise ValueError("clock_metadata keys must be exactly " + ", ".join(sorted(_CLOCK_KEYS)))
    if not _text(clock_metadata["clock_id"]):
        raise ValueError("clock_id must be nonempty text")
    if clock_metadata["clock_kind"] != "monotonic":
        raise ValueError("clock_kind must be monotonic; UTC is a correlation anchor only")
    if clock_metadata["clock_scope"] not in ("host", "process"):
        raise ValueError("clock_scope must be host or process")
    anchor = clock_metadata["utc_anchor"]
    if anchor is not None and not _text(anchor):
        raise ValueError("utc_anchor must be text or None")


def _check_allocation(allocation):
    if type(allocation) is not dict or set(allocation) != _ALLOCATION_KEYS:
        return "allocation keys must be exactly " + ", ".join(sorted(_ALLOCATION_KEYS))
    if not _text(allocation["allocation_id"]) or not _text(allocation["clock_id"]):
        return "allocation_id and clock_id must be nonempty text"
    count, devices = allocation["gpu_count"], allocation["devices"]
    if not _int(count) or count < 1:
        return "gpu_count must be a positive integer"
    if (
        type(devices) is not list
        or len(devices) != count
        or not all(_text(d) for d in devices)
        or len(set(devices)) != count
    ):
        return "devices must list gpu_count distinct device IDs"
    start, end = allocation["billed_start_us"], allocation["billed_end_us"]
    if not _int(start) or not _int(end) or start < 0 or end <= start:
        return "billed_start_us < billed_end_us must be non-negative integers"
    segments = allocation["segments"]
    if type(segments) is not list or not segments:
        return "segments must be a nonempty list"
    for segment in segments:
        if (
            type(segment) is not dict
            or set(segment) != {"phase", "start_us", "end_us"}
            or segment["phase"] not in PHASES
            or not _int(segment["start_us"])
            or not _int(segment["end_us"])
            or segment["start_us"] >= segment["end_us"]
        ):
            return f"segments need phase in {PHASES} and integer start_us < end_us"
    return None


def _partition(allocation, issues):
    start, end = allocation["billed_start_us"], allocation["billed_end_us"]
    phases = {phase: 0 for phase in PHASES}
    unaccounted = []
    status = "passed"
    cursor = start
    for position, segment in enumerate(
        sorted(allocation["segments"], key=lambda s: (s["start_us"], s["end_us"], s["phase"]))
    ):
        a, b = segment["start_us"], segment["end_us"]
        path = f"allocation/segments/{position}"
        if a < start or b > end:
            issues.append(_issue("partition_out_of_bounds", path, "Segment outside billed time"))
            status = "blocked"
        if a < cursor:
            issues.append(_issue("partition_overlap", path, "Segments overlap"))
            status = "blocked"
        elif a > cursor:
            unaccounted.append([cursor, a])
        cursor = max(cursor, b)
        phases[segment["phase"]] += min(b, end) - max(a, start) if b > start and a < end else 0
    if cursor < end:
        unaccounted.append([cursor, end])
    if unaccounted:
        issues.append(
            _issue("partition_gap", "allocation/segments", "Billed time not fully partitioned")
        )
        status = "blocked"
    billed = end - start
    count = allocation["gpu_count"]
    return {
        "status": status,
        "phases": {
            phase: {
                "us": us,
                "share": _quantity(Fraction(us, billed)),
                "gpu_hours": _quantity(Fraction(us * count, US_PER_HOUR)),
            }
            for phase, us in phases.items()
        },
        "unaccounted_us": sum(b - a for a, b in unaccounted),
        "unaccounted_intervals_us": unaccounted,
    }


def _check_sample(row):
    if type(row) is not dict or set(row) != _SAMPLE_KEYS:
        return "sample keys must be exactly " + ", ".join(sorted(_SAMPLE_KEYS))
    if not _text(row["metric"]) or not _text(row["device"]) or not _text(row["raw_ref"]):
        return "metric, device and raw_ref must be nonempty text"
    if not _text(row["clock_id"]) or not _int(row["t_us"]):
        return "clock_id must be text and t_us an integer"
    if row["status"] not in SAMPLE_STATUSES:
        return f"status must be one of {SAMPLE_STATUSES}; a missing status is not ok"
    if row["status"] == "ok":
        if type(row["value"]) is bool or not _finite(row["value"]):
            return "an ok sample needs a finite numeric value"
    elif row["value"] is not None:
        return "a non-ok sample must carry value None"
    return None


def _check_rollout(row):
    if type(row) is not dict or set(row) != _ROLLOUT_KEYS:
        return "rollout keys must be exactly " + ", ".join(sorted(_ROLLOUT_KEYS))
    if not all(_text(row[k]) for k in ("attempt_id", "job_id", "clock_id")):
        return "attempt_id, job_id and clock_id must be nonempty text"
    if row["status"] not in ROLLOUT_STATUSES:
        return f"status must be one of {ROLLOUT_STATUSES}"
    if row["useful"] not in (True, False, None) or (
        row["useful"] is not None and type(row["useful"]) is not bool
    ):
        return "useful must be True, False or None"
    if row["useful"] is True and row["status"] != "completed":
        return "only a completed attempt can be a useful episode"
    if not _int(row["start_us"]):
        return "start_us must be an integer"
    if row["end_us"] is None:
        if row["status"] != "unresolved":
            return "end_us may be None only for an unresolved attempt"
    elif not _int(row["end_us"]) or row["end_us"] < row["start_us"]:
        return "end_us must be an integer no earlier than start_us"
    return None


def _union_us(intervals):
    total, cursor = 0, None
    for a, b in sorted(intervals):
        if cursor is None or a > cursor:
            total += b - a
            cursor = b
        elif b > cursor:
            total += b - cursor
            cursor = b
    return total


def _series_summary(metric, rows, window, max_gap):
    unit, kind, _ = PINNED_METRICS[metric]
    counts = {status: 0 for status in SAMPLE_STATUSES}
    for row in rows:
        counts[row["status"]] += 1
    ok = sorted((row["t_us"], row["value"]) for row in rows if row["status"] == "ok")
    start, end = window
    summary = {
        "unit": unit,
        "kind": kind,
        "n_samples": len(rows),
        "status_counts": counts,
        "availability": None,
        "first_ok_us": ok[0][0] if ok else None,
        "last_ok_us": ok[-1][0] if ok else None,
        "max_observed_gap_us": None,
        "gaps_over_limit_us": [],
        "raw_refs": sorted({row["raw_ref"] for row in rows}),
    }
    if not rows:
        summary["availability"] = "absent"
    elif not ok:
        statuses = {row["status"] for row in rows}
        if statuses == {"not_supported"}:
            summary["availability"] = "unsupported"
        elif statuses == {"no_data"}:
            summary["availability"] = "no_data"
        else:
            summary["availability"] = "failed"
    else:
        times = [start] + [t for t, _ in ok] + [end]
        gaps = list(itertools.pairwise(times))
        summary["max_observed_gap_us"] = max(b - a for a, b in gaps)
        summary["gaps_over_limit_us"] = [[a, b] for a, b in gaps if b - a > max_gap]
        summary["availability"] = "incomplete" if summary["gaps_over_limit_us"] else "healthy"
    if kind == "gauge" and ok:
        top = max(value for _, value in ok)
        summary["gauge"] = {
            "min": min(value for _, value in ok),
            "max": top,
            "max_t_us": next(t for t, value in ok if value == top),
        }
    if kind == "counter":
        summary["counter"] = _counter(ok, max_gap)
    return summary, ok


def _counter(ok, max_gap):
    if len(ok) < 2:
        return {
            "resets": 0,
            "reset_events_us": [],
            "increase": _quantity(reason="fewer_than_two_ok_samples"),
            "increase_lower_bound": _quantity(reason="fewer_than_two_ok_samples"),
        }
    increase = Fraction(0)
    resets = []
    for (_, previous), (t, value) in itertools.pairwise(ok):
        if value < previous:
            resets.append(t)
            increase += Fraction(value)
        else:
            increase += Fraction(value) - Fraction(previous)
    # A reset hidden in a gap over the limit is undetectable: the increase across it
    # is a lower bound only, matching counter-delta energy.
    long_gap = any(t1 - t0 > max_gap for (t0, _), (t1, _) in itertools.pairwise(ok))
    if resets:
        total = _quantity(reason="counter_reset", status="unverified")
    elif long_gap:
        total = _quantity(reason="gap_beyond_coverage_limit", status="unverified")
    else:
        total = _quantity(increase)
    return {
        "resets": len(resets),
        "reset_events_us": resets,
        "increase": total,
        "increase_lower_bound": _quantity(increase),
    }


def _power_energy(ok, window, max_gap):
    start, end = window
    if not ok:
        return {
            "status": "not_estimable",
            "total_energy_j": _quantity(reason="no_ok_samples"),
            "covered_energy_j": _quantity(Fraction(0)),
            "covered_us": 0,
            "uncovered_intervals_us": [[start, end]],
        }
    energy = Fraction(0)
    covered = 0
    uncovered = []
    if ok[0][0] > start:
        uncovered.append([start, ok[0][0]])
    for (t0, p0), (t1, p1) in itertools.pairwise(ok):
        if t1 - t0 > max_gap:
            uncovered.append([t0, t1])
            continue
        energy += (Fraction(p0) + Fraction(p1)) / 2 * Fraction(t1 - t0, US_PER_SECOND)
        covered += t1 - t0
    if ok[-1][0] < end:
        uncovered.append([ok[-1][0], end])
    complete = not uncovered
    return {
        "status": "passed" if complete else "not_estimable",
        "total_energy_j": _quantity(energy) if complete else _quantity(reason="incomplete_coverage"),
        "covered_energy_j": _quantity(energy),
        "covered_us": covered,
        "uncovered_intervals_us": uncovered,
    }


def _counter_energy(ok, window, max_gap):
    start, end = window
    counter = _counter(ok, max_gap)
    edges_covered = bool(ok) and ok[0][0] == start and ok[-1][0] == end
    # A reset inside a gap longer than the limit is undetectable from samples, so
    # the increase across it is only a lower bound, never a passed total.
    long_gaps = [[t0, t1] for (t0, _), (t1, _) in itertools.pairwise(ok) if t1 - t0 > max_gap]
    lower = counter["increase_lower_bound"]
    lower_j = _quantity(Fraction(lower["exact"]) / 1000) if lower["exact"] is not None else lower
    if counter["resets"]:
        status, reason = "unverified", "counter_reset"
    elif not edges_covered or lower["exact"] is None:
        status, reason = "not_estimable", "incomplete_coverage"
    elif long_gaps:
        status, reason = "unverified", "gap_beyond_coverage_limit"
    else:
        status, reason = "passed", None
    return {
        "status": status,
        "total_energy_j": lower_j
        if status == "passed"
        else _quantity(reason=reason, status="unverified" if status == "unverified" else None),
        "lower_bound_energy_j": lower_j,
        "reset_events_us": counter["reset_events_us"],
        "gaps_over_limit_us": long_gaps,
        "covered_interval_us": [ok[0][0], ok[-1][0]] if ok else None,
        "unit_conversion": "mJ / 1000 = J",
    }


def summarize_telemetry(
    samples,
    allocation,
    rollouts,
    *,
    clock_metadata,
    max_sample_gap_us,
    scope,
    stabilization_rule=None,
):
    """Summarize paid time, useful work, coverage, energy, memory and XIDs (L1).

    ``status`` is ``passed`` when every input is well formed and the partition
    sums exactly to billed time; it describes summary consistency only and is
    never host qualification or rental readiness.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if not _int(max_sample_gap_us) or max_sample_gap_us <= 0:
        raise ValueError("max_sample_gap_us must be a positive integer")
    _check_clock(clock_metadata)
    if type(samples) is not list or type(rollouts) is not list:
        raise ValueError("samples and rollouts must be lists")

    issues = []
    problem = _check_allocation(allocation)
    alloc_ok = problem is None
    if not alloc_ok:
        issues.append(_issue("allocation_invalid", "allocation", problem))
    clock_id = clock_metadata["clock_id"]
    if alloc_ok and allocation["clock_id"] != clock_id:
        issues.append(
            _issue("clock_scope_mismatch", "allocation/clock_id", "Allocation uses another clock")
        )
    window = (allocation["billed_start_us"], allocation["billed_end_us"]) if alloc_ok else None
    devices = sorted(allocation["devices"]) if alloc_ok else []

    accepted = []
    dropped_by_reason = {}
    dropped_series = {}  # (pinned metric, allocated device) -> samples dropped outside billing
    for position, row in enumerate(samples):
        path = f"samples/{position}"
        problem = _check_sample(row)
        if problem:
            issue = _issue("sample_invalid", path, problem)
        elif row["clock_id"] != clock_id:
            issue = _issue("clock_scope_mismatch", path, "Sample uses another clock")
        elif row["metric"] not in PINNED_METRICS:
            issue = _issue(
                "metric_not_pinned", path, f"Unpinned metric {row['metric']!r}", "warning"
            )
        elif alloc_ok and row["device"] not in devices:
            issue = _issue("device_not_allocated", path, "Device not in the allocation")
        elif alloc_ok and not window[0] <= row["t_us"] <= window[1]:
            issue = _issue(
                "sample_outside_allocation", path, "Sample outside billed time", "warning"
            )
            series = (row["metric"], row["device"])
            dropped_series[series] = dropped_series.get(series, 0) + 1
        else:
            accepted.append(row)
            continue
        issues.append(issue)
        dropped_by_reason[issue["code"]] = dropped_by_reason.get(issue["code"], 0) + 1
    keyed = {}
    for row in accepted:
        keyed.setdefault((row["metric"], row["device"], row["t_us"]), []).append(row)
    duplicated = {key for key, rows in keyed.items() if len(rows) > 1}
    for key in sorted(duplicated):
        issues.append(
            _issue("duplicate_sample", f"samples/{key[0]}/{key[1]}/{key[2]}",
                   "More than one sample for one metric, device and time")
        )
    before_dedup = len(accepted)
    accepted = [r for r in accepted if (r["metric"], r["device"], r["t_us"]) not in duplicated]
    if before_dedup != len(accepted):
        dropped_by_reason["duplicate_sample"] = before_dedup - len(accepted)
    if samples and not accepted:
        # Supplied-but-discarded telemetry must not read as "never supplied".
        issues.append(
            _issue("all_samples_dropped", "samples", "Every supplied sample was dropped")
        )

    attempts = []
    for position, row in enumerate(rollouts):
        path = f"rollouts/{position}"
        problem = _check_rollout(row)
        if problem:
            issues.append(_issue("rollout_invalid", path, problem))
        elif row["clock_id"] != clock_id:
            issues.append(_issue("clock_scope_mismatch", path, "Attempt uses another clock"))
        elif alloc_ok and (
            row["start_us"] < window[0]
            or (row["end_us"] if row["end_us"] is not None else row["start_us"]) > window[1]
        ):
            issues.append(
                _issue("rollout_outside_allocation", path, "Attempt outside billed time")
            )
        else:
            attempts.append(row)
    attempt_ids = {}
    for row in attempts:
        attempt_ids[row["attempt_id"]] = attempt_ids.get(row["attempt_id"], 0) + 1
    for attempt in sorted(a for a, n in attempt_ids.items() if n > 1):
        issues.append(_issue("duplicate_attempt", f"rollouts/{attempt}", "Attempt ID repeated"))
    attempts = [row for row in attempts if attempt_ids[row["attempt_id"]] == 1]
    rejected_attempts = len(rollouts) - len(attempts)

    allocation_section = {
        "allocation_id": allocation["allocation_id"] if alloc_ok else None,
        "gpu_count": allocation["gpu_count"] if alloc_ok else None,
        "devices": devices,
        "billed_us": None,
        "billed_hours": _quantity(reason="allocation_invalid"),
        "gpu_hours": _quantity(reason="allocation_invalid"),
        "partition": {"status": "blocked", "phases": None, "unaccounted_us": None,
                      "unaccounted_intervals_us": None},
    }
    billed_hours = gpu_hours = None
    if alloc_ok:
        billed = window[1] - window[0]
        billed_hours = Fraction(billed, US_PER_HOUR)
        gpu_hours = billed_hours * allocation["gpu_count"]
        allocation_section.update(
            billed_us=billed,
            billed_hours=_quantity(billed_hours),
            gpu_hours=_quantity(gpu_hours),
            partition=_partition(allocation, issues),
        )

    useful = sum(1 for row in attempts if row["useful"] is True)
    unknown = sum(1 for row in attempts if row["useful"] is None)
    by_status = {status: 0 for status in ROLLOUT_STATUSES}
    for row in attempts:
        by_status[row["status"]] += 1
    useful_work = {
        "useful_episodes": useful,
        "useful_unknown": unknown,
        "attempts_by_status": by_status,
        "per_wall_hour": _quantity(Fraction(useful) / billed_hours)
        if billed_hours
        else _quantity(reason="allocation_invalid"),
        "per_gpu_hour": _quantity(Fraction(useful) / gpu_hours)
        if gpu_hours
        else _quantity(reason="allocation_invalid"),
        "rejected_attempts": rejected_attempts,
        # Dropping an attempt can only lower the useful count over fixed billed time.
        "rates_are_lower_bounds": unknown > 0 or rejected_attempts > 0,
        "status": "blocked"
        if rejected_attempts
        else ("unverified" if unknown else ("passed" if alloc_ok else "not_estimable")),
        "definition": "attempts with useful=True per billed wall-hour and per billed GPU-hour; "
        "the flag is consumed as supplied (its producer's rule is an agent technical default "
        "(D-018), not the scientific definition); exit codes are never read",
    }
    closed = [(row["start_us"], row["end_us"]) for row in attempts if row["end_us"] is not None]
    service_time = {
        "rollout_service_us_sum": sum(b - a for a, b in closed),
        "rollout_service_union_us": _union_us(closed),
        "note": "service intervals may overlap; never summed into billed time",
    }

    metrics, energy_power, energy_counter, memory, xid = {}, {}, {}, {}, {}
    for metric in sorted(PINNED_METRICS):
        role = PINNED_METRICS[metric][2]
        metrics[metric] = {}
        for device in devices:
            rows = [r for r in accepted if r["metric"] == metric and r["device"] == device]
            summary, ok = _series_summary(metric, rows, window, max_sample_gap_us)
            summary["dropped_samples"] = dropped_series.get((metric, device), 0)
            if not rows and summary["dropped_samples"]:
                summary["availability"] = "dropped"
            metrics[metric][device] = summary
            if role == "power":
                energy_power.setdefault(metric, {})[device] = _power_energy(
                    ok, window, max_sample_gap_us
                )
            elif role == "energy":
                energy_counter.setdefault(metric, {})[device] = _counter_energy(
                    ok, window, max_sample_gap_us
                )
            elif role == "memory":
                memory.setdefault(metric, {})[device] = {
                    "high_water_mib": summary["gauge"]["max"] if ok else None,
                    "t_us": summary["gauge"]["max_t_us"] if ok else None,
                    "availability": summary["availability"],
                    "max_observed_gap_us": summary["max_observed_gap_us"],
                    "limitation": "maximum of ok samples only; peaks between samples are "
                    "unobserved",
                }
            elif role == "xid":
                xid[device] = {
                    "metric": metric,
                    "availability": summary["availability"],
                    "observed_values": sorted({value for _, value in ok}),
                    "error_count": None,
                    "fault_observability": "unverified",
                    "note": "DCGM reports the last XID value per reading; NO_DATA, absent "
                    "or unsupported readings are not zero XIDs and no count is derived",
                }

    if stabilization_rule is None:
        stabilized = {"status": "blocked", "code": "stabilization_rule_unregistered",
                      "value": None}
    else:
        stabilized = {"status": "blocked", "code": "unsupported_stabilization_rule",
                      "value": None}

    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    blocked = any(item["severity"] == "error" for item in issues)
    samples_hash, samples_non_json = _supplied_hash(samples)
    rollouts_hash, rollouts_non_json = _supplied_hash(rollouts)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "gate.s2_telemetry", "source_sha256": _source_sha256()},
        "input_hashes": {
            "samples": samples_hash,
            "accepted_samples": _supplied_hash(accepted)[0],
            "allocation": _safe_hash(allocation),
            "rollouts": rollouts_hash,
            "accepted_rollouts": _supplied_hash(attempts)[0],
            "stabilization_rule": None
            if stabilization_rule is None
            else _safe_hash(stabilization_rule),
        },
        "units": {
            "time": "integer microseconds on the declared monotonic clock",
            "hours": "billed hours (exact rational and float)",
            "energy": "J",
            "metrics": {metric: spec[0] for metric, spec in sorted(PINNED_METRICS.items())},
        },
        "status": "blocked" if blocked else "passed",
        "status_meaning": "summary consistency only; never host qualification or rental "
        "readiness",
        "clock": {
            "clock_id": clock_id,
            "clock_kind": clock_metadata["clock_kind"],
            "clock_scope": clock_metadata["clock_scope"],
            "utc_anchor": clock_metadata["utc_anchor"],
        },
        "max_sample_gap_us": max_sample_gap_us,
        "samples": {
            "supplied": len(samples),
            "accepted": len(accepted),
            "non_json": samples_non_json,
            "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        },
        "rollouts": {"supplied": len(rollouts), "non_json": rollouts_non_json},
        "allocation": allocation_section,
        "useful_work": useful_work,
        "stabilized_throughput": stabilized,
        "service_time": service_time,
        "metrics": metrics,
        "energy": {"power_integration": energy_power, "counter_delta": energy_counter},
        "memory": memory,
        "xid": xid,
        "device_latency": {
            "status": "not_applicable",
            "reason": "no synchronized device timing input; telemetry samples cannot "
            "support a kernel or device latency claim",
        },
        "host_qualification": "qualify_on_host",
        "issues": issues,
    }


def _supplied_hash(rows):
    """Order-invariant hash of every supplied row, accepted or not.

    Each row contributes its canonical-JSON identity hash; a row that is not
    JSON-native contributes a marker naming its position. Returns
    ``(hash, number of non-JSON rows)``.
    """
    digests, non_json = [], 0
    for position, row in enumerate(rows):
        try:
            digests.append(identity_hash(row))
        except (ValueError, TypeError):
            non_json += 1
            digests.append(identity_hash({"non_json_row_position": position}))
    return identity_hash(sorted(digests)), non_json


def _safe_hash(value):
    try:
        return identity_hash(value)
    except ValueError:
        return None
