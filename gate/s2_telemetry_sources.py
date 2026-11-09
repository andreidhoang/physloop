"""S2 telemetry producers: DCGM scrape parser, allocation partition and useful attempt rows.

Readiness CPU-07 / S2.7 [C] (G7); closes the completeness-audit gap "CPU-07/S2.7
producers" (nothing parsed DCGM exporter output or produced the allocation segment
partition and useful-flagged attempt rows ``gate.s2_telemetry.summarize_telemetry``
consumes). Agent-authored evaluator engineering under D-018/D-024. Every output is an
input to ``summarize_telemetry``; this module computes no throughput, energy or verdict.

Functions and verification ceiling (build-spec section 1):

- ``parse_dcgm_prom`` - L1 over literal Prometheus text; L2 for the exporter format:
  metric names, ``# HELP`` text, ``# TYPE`` kinds and the ``gpu``/``UUID`` labels are
  pinned to the dcgm-exporter scrapes recorded by ``scripts/vast_exporter_control.sh``
  (pinned image ``nvcr.io/nvidia/k8s/dcgm-exporter@sha256:804c17f6...``, embedded DCGM
  4.6.1 / exporter 4.8.4, ``-f /etc/dcgm-exporter/default-counters.csv``) in
  ``evidence/s2-vast-host-20261101-v1/exporter-v1/sample-*.prom``, copied byte-identically
  with SHA-256 into ``tests/fixtures/s2_telemetry_sources/``.
  ``tests/test_vast_monitor_reuse.py`` fakes that probe's curl with a bare
  ``DCGM_FI_DEV_GPU_UTIL 80`` line; the probe accepts any body, whereas this parser
  rejects a pinned metric without its recorded HELP/TYPE (``pinned_metadata_missing``).
  ``DCGM_FI_DEV_XID_ERRORS`` never appeared in a recorded scrape
  (``exporter-metrics-summary.json``: ``[]``). F6-05: its HELP and TYPE are pinned from the
  dcgm-exporter default counters CSV vendored byte-identically, with its Apache-2.0 LICENSE
  and provenance (tag ``4.8.4``, commit ``fafd151``, git blob ``6e6de4af``, SHA-256
  ``ddf9d12e...``), into ``tests/fixtures/s2_telemetry_sources/dcgm-exporter/``: release
  4.8.4 because the recorded ``exporter.log`` says ``Version=4.8.4`` and reads
  ``/etc/dcgm-exporter/default-counters.csv``. The pin is L2: the image's embedded CSV was
  not captured, but every HELP/TYPE pair of the recorded scrapes equals that CSV's trimmed
  columns (``tests/test_s2_telemetry_sources.py``). The literal lives here; this module never
  reads the fixture. A pinned metric whose HELP is not pinned (``help`` ``None``) still
  yields a ``help_not_pinned`` warning and an ``unverified`` scrape, never ``passed``. XID
  event delivery on a host stays L3 (``qualify_on_host``). A body
  with no ``DCGM_`` sample line (blank, comment-only) is a failed scrape
  (``scrape_no_samples``), never an empty pass. Every pinned metric is non-negative: a
  negative gauge or counter is ``scrape_failed``. Delivery on an admitted host is L3
  (``qualify_on_host``).
- ``parse_dcgm_scrapes`` - L1: a time-ordered scrape series; counter resets (a decrease
  between consecutive ``ok`` readings, the Prometheus convention ``summarize_telemetry``
  also applies) and failed scrapes are reported, never repaired.
- ``useful_attempt_rows`` - L1 over ``gate.s2_ledger.reconcile_attempts`` output and
  ``scripts/s2_run_units.py`` unit records (``_Run._envelope``; schema version 1). The
  report must re-derive byte-identically (canonical JSON) from its own ``schedule`` and
  ``events``; inputs that cannot be canonically hashed raise ``ValueError``. F6 residual
  (d): ``reconcile_attempts`` stores a non-JSON input value (a NaN ``timestamp_us``) as
  ``null`` with an ``invalid_json_value`` issue, which re-derivation from the stored values
  cannot reproduce. Such a report is accepted only when it equals the re-derivation after
  removing ``invalid_json_value`` issues and the admission they caused (``admitted`` and
  ``selected_attempt_id``); it is then ``blocked`` (``ledger_input_non_json``) with every
  flag ``None``, never raised and never useful. A non-list events container (a tuple) is
  stored as ``events: null`` with that issue at path ``events``; only then may ``events`` be
  null, and the report is ``blocked`` with no rows.
- ``allocation_from_journals`` - L1 over ``gate.s2_execution.LifecycleJournal.read``
  records and the runner's ``ledger-events.jsonl`` (event schema v1/v2). Each attempt row
  is bound to the journal's own events: integer times, ``status`` equal to the terminal
  event type (``unresolved`` iff no terminal), ``useful=True`` only on a ``completed``
  terminal ``terminal_established`` accepts and ``useful=False`` only on ``failed`` or
  ``cancelled``; any contradiction raises ``ValueError``. The clock alignment it needs is
  L3 until the runner records an anchor (see limitations).

Interface reconciliation (recorded, not silently chosen). The coordinator brief names
``parse_dcgm_prom(text, *, scrape_t_us, device_map)``,
``allocation_from_journals(lifecycle_journal_records, runner_journal_records, *,
billed_interval)`` and ``useful_attempt_rows(ledger_report, unit_records)``. Required
keywords are added because the consumer schema needs them and no default may be invented:
``device_key``, ``clock_id``, ``scrape_ref`` and ``scope`` for the parser;
``clock_anchor``, ``attempt_rows`` and ``scope`` for the partition; ``clock_id`` and
``scope`` for the rows. ``billed_interval`` carries the allocation identity and devices
``summarize_telemetry`` requires.

Partition phases. The brief's phases are not ``gate.s2_telemetry.PHASES``; the
partition keeps the brief's names and ``telemetry_allocation`` maps them
(``PHASE_MAP``, agent technical default D-018): provisioning -> acquisition, bootstrap ->
startup, unit_startup -> warm_up, useful_rollout -> active, retry_or_failed -> retry,
idle -> idle, export -> export, teardown -> tail. ``unknown`` has no consumer phase and
is left as a gap, so ``summarize_telemetry`` reports ``partition_gap`` (blocked) with
``unaccounted_us`` equal to the unknown total; unknown time is never assigned to useful.

Lifecycle state -> phase (agent technical default D-018): admitted, create_intent,
create_requested, uncertain_create, bound and armed are provisioning; bootstrapping is
bootstrap; running is resolved by the runner journal; exporting and export_verified are
export; arm_failed, cutoff_reached, stopping, stopped and destroying are teardown;
billed time after absent_verified/reconciled, in escalated or aborted, or before the
first lifecycle record is unknown. A state's phase lasts from its transition record's
``at_utc`` to the next phase change; ``at_utc`` is wall time stored beside, not inside,
the hash-chained body, so the chain proves record order and content but not their times.

Runner journal (agent technical defaults D-018). A unit spans its job's first ledger
event (the ``started`` event, emitted before any side effect) to its last terminal event
(emitted after capture, teardown and reconcile). The pinned runner records no per-step
times, so a unit's own preparation, driver start, capture and teardown are inside its
interval and cannot be separated: the whole interval is useful_rollout when every
attempt is useful, retry_or_failed when every attempt is not useful, and unknown when any
attempt is unclassified (``unit_classification_unresolved``) or has no terminal event
(``attempt_unresolved``, until the next runner event or the end of the running phase).
Running time before the first unit is unit_startup (runner start, validation, the first
budget check); time between and after units is idle; a running phase with no runner
records is unknown. Lifecycle times reach the runner's monotonic clock through
``clock_anchor``: ``t = monotonic_us + (at_utc - utc)``; every lifecycle phase change
gets an unknown band of +-``uncertainty_us``. Runner events outside the running phase
(beyond that band) are an error and the time they cover outside it is unknown.

Useful flags (agent technical default (D-018), the coordinator's rule "derived only from
established completion"; ``gate.s2_telemetry`` consumes the flag under the same label):
``True`` only for a ``completed`` attempt whose terminal event
``gate.s2_executor.terminal_established`` accepts (v2 ``process_completed``), that is
its job's only attempt, with exactly one consistent unit record. A failed or cancelled
attempt is ``False``. Everything else is ``None`` with a reason: a completed attempt in a
retried job waits for Huy's retry-selection rule (``retry_selection_unregistered``); a v1
``exact_horizon`` completion is ``completion_not_established``; an incomplete attempt is
``attempt_incomplete``; any ledger issue outside ``OUTCOME_ISSUE_CODES`` (including an
unknown code) impugns the evidence and makes every flag ``None``
(``ledger_integrity_issue``). Outcome issues on one job never change another job's flag.
This is useful-work accounting only; scientific admission (validity, horizon, retry
selection, missing data) stays Huy's and downstream.

Owner inputs left blank: retry-selection rule, stabilization rule, the scientific
definition of a useful episode beyond established completion, and any tolerance on the
clock anchor's uncertainty.

This module never reads files (except its own source, for the producer hash), runs
subprocesses or contacts a network (no curl, DCGM,
nvidia-smi or provider call), never fills a gap or reads a missing value as zero, never
turns an absent XID series into zero XIDs, never infers success from an exit code, never
stores scrape text (only its SHA-256), and never imports protected inference.
"""

import bisect
import hashlib
import itertools
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gate.s2_execution import STATES as LIFECYCLE_STATES
from gate.s2_executor import terminal_established
from gate.s2_identity import identity_hash
from gate.s2_ledger import _valid_event, reconcile_attempts
from gate.s2_telemetry import PINNED_METRICS, ROLLOUT_STATUSES, SCOPES

SCHEMA_VERSION = 1
SCRAPE_KIND = "s2_dcgm_scrape"
SCRAPES_KIND = "s2_dcgm_scrapes"
PARTITION_KIND = "s2_allocation_partition"
ROWS_KIND = "s2_useful_attempt_rows"
DEVICE_KEYS = ("UUID", "gpu")

# HELP text and TYPE exactly as recorded in evidence/s2-vast-host-20261101-v1/exporter-v1
# (XID: the vendored dcgm-exporter 4.8.4 etc/default-counters.csv; see the module docstring).
DCGM_PINNED = {
    "DCGM_FI_DEV_FB_USED": {"help": "Framebuffer memory used (in MiB).", "type": "gauge"},
    "DCGM_FI_DEV_GPU_UTIL": {"help": "GPU utilization (in %).", "type": "gauge"},
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": {"help": "Total number of PCIe retries.",
                                        "type": "counter"},
    "DCGM_FI_DEV_POWER_USAGE": {"help": "Power draw (in W).", "type": "gauge"},
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": {
        "help": "Total energy consumption since the driver was last reloaded (in mJ).",
        "type": "counter"},
    # Never in a recorded scrape: pinned from the vendored exporter 4.8.4 default counters CSV.
    "DCGM_FI_DEV_XID_ERRORS": {"help": "Value of the last XID error encountered.",
                               "type": "gauge"},
}
_DCGM_ORDER = sorted(DCGM_PINNED)
if set(DCGM_PINNED) != {m for m in PINNED_METRICS if m.startswith("DCGM_")}:
    raise ImportError("DCGM_PINNED must cover exactly the DCGM metrics s2_telemetry pins")

PHASE_MAP = {
    "provisioning": "acquisition",
    "bootstrap": "startup",
    "unit_startup": "warm_up",
    "useful_rollout": "active",
    "retry_or_failed": "retry",
    "idle": "idle",
    "export": "export",
    "teardown": "tail",
}
SEGMENT_PHASES = (*PHASE_MAP, "unknown")
_AFTER_ABSENCE = ("unknown", "billed_after_verified_absence")
_LIFECYCLE = {
    **dict.fromkeys(("admitted", "create_intent", "create_requested", "uncertain_create",
                     "bound", "armed"), ("provisioning", None)),
    "bootstrapping": ("bootstrap", None),
    "running": ("running", None),
    "exporting": ("export", None),
    "export_verified": ("export", None),
    **dict.fromkeys(("arm_failed", "cutoff_reached", "stopping", "stopped", "destroying"),
                    ("teardown", None)),
    "absent_verified": _AFTER_ABSENCE,
    "reconciled": _AFTER_ABSENCE,
    "escalated": ("unknown", "lifecycle_escalated"),
    "aborted": ("unknown", "lifecycle_aborted"),
}
if set(_LIFECYCLE) != set(LIFECYCLE_STATES):
    raise ImportError("every gate.s2_execution state needs a partition phase")

# reconcile_attempts codes that describe one job's outcome; every other code (including
# one this module does not know) impugns the ledger evidence.
OUTCOME_ISSUE_CODES = frozenset({"job_not_completed", "missing_job", "incomplete_attempt",
                                 "retry_selection_unregistered",
                                 "multiple_completed_attempts"})
_ROW_KEYS = {"attempt_id", "job_id", "clock_id", "start_us", "end_us", "status", "useful"}
_REPORT_KEYS = {"admitted", "issues", "jobs", "attempts", "counts", "schedule", "events"}
NON_JSON_CODE = "invalid_json_value"  # gate.s2_ledger.reconcile_attempts snapshot()
USEFUL_DEFINITION = (
    "agent technical default (D-018): useful=True only for an established completion "
    "(terminal_established) that is its job's only attempt with one consistent unit record; "
    "failed or cancelled is False; anything else is None with a reason. The scientific "
    "definition of a useful episode stays Huy's")
_ATTEMPT_KEYS = {"job_id", "attempt_id", "parent_attempt_id", "retry_reason",
                 "upstream_identity", "state", "started_timestamp_us",
                 "terminal_timestamp_us", "terminal_reason", "events"}
_STATE_STATUS = {"completed": "completed", "failed": "failed", "cancelled": "cancelled",
                 "incomplete": "unresolved"}
# scripts/s2_run_units.py ``_Run._envelope`` (UNIT_RECORD_KIND, UNIT_RECORD_VERSION).
UNIT_RECORD_KIND = "s2_unit_record"
UNIT_RECORD_VERSION = 1
_UNIT_RECORD_KEYS = {"artifact_kind", "schema_version", "scope", "producer", "input_hashes",
                     "units", "status", "status_reasons", "index", "job_id", "scene_id",
                     "steps", "outcome", "stop_run", "compose_exit_code", "unit_outcome",
                     "ledger_events", "issues"}
_JOURNAL_KEYS = {"seq", "prev_hash", "payload", "record_hash", "at_utc"}
_BILLED_KEYS = {"allocation_id", "clock_id", "gpu_count", "devices", "start_us", "end_us",
                "source_ref"}
_ANCHOR_KEYS = {"clock_id", "monotonic_us", "utc", "uncertainty_us", "anchor_ref"}
_GENESIS = "0" * 64
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_US = timedelta(microseconds=1)

_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_TIMESTAMP = re.compile(r"-?[0-9]+")
_PROM_TYPES = ("counter", "gauge", "histogram", "summary", "untyped")
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _producer():
    return {"module": "gate.s2_telemetry_sources", "source_sha256": _source_sha256()}


def _issue(code, path, message, severity="error"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _sorted_issues(issues):
    return sorted(issues, key=lambda item: (item["path"], item["code"], item["message"]))


def _status(issues, unresolved):
    if any(item["severity"] == "error" for item in issues):
        return "blocked"
    return "unverified" if unresolved else "passed"


def _text(value):
    return type(value) is str and bool(value.strip()) and "\x00" not in value


def _int(value):
    return type(value) is int


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_scope(scope):
    _require(scope in SCOPES, f"scope must be one of {SCOPES}")


# ------------------------------------------------------------------ DCGM exposition


class _Malformed(Exception):
    def __init__(self, code, line, message):
        super().__init__(message)
        self.code, self.line, self.message = code, line, message


def _parse_labels(rest, line_no):
    """Parse ``name="value",...}``; returns (labels, text after ``}``)."""
    labels, i = {}, 0
    while True:
        if rest[i:i + 1] == "}":
            return labels, rest[i + 1:]
        match = _LABEL_NAME.match(rest, i)
        if not match or rest[match.end():match.end() + 2] != '="':
            raise _Malformed("malformed_line", line_no, "label must be name=\"value\"")
        name, i, value = match.group(), match.end() + 2, []
        while True:
            if i >= len(rest):
                raise _Malformed("malformed_line", line_no, "unterminated label value")
            char = rest[i]
            if char == "\\":
                escaped = _ESCAPES.get(rest[i + 1:i + 2])
                if escaped is None:
                    raise _Malformed("malformed_line", line_no, "invalid label escape")
                value.append(escaped)
                i += 2
            elif char == '"':
                i += 1
                break
            else:
                value.append(char)
                i += 1
        if name in labels:
            raise _Malformed("malformed_line", line_no, f"label {name!r} repeated")
        labels[name] = "".join(value)
        if rest[i:i + 1] == ",":
            i += 1
        elif rest[i:i + 1] != "}":
            raise _Malformed("malformed_line", line_no, "labels must be comma separated")


def _parse_value(token, line_no):
    """(kind, value): ``number`` with an int or finite float, ``nan`` or ``inf``."""
    if token == "NaN":
        return "nan", None
    if token in ("+Inf", "-Inf"):
        return "inf", None
    if _INTEGER.fullmatch(token):
        number = int(token)
        try:
            float(number)
        except OverflowError:
            return "inf", None
        return "number", number
    if _NUMBER.fullmatch(token):
        number = float(token)
        return ("number", number) if math.isfinite(number) else ("inf", None)
    raise _Malformed("malformed_line", line_no, "value is not a Prometheus number")


def _unescape_help(text, line_no):
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\":
            escaped = {"\\": "\\", "n": "\n"}.get(text[i + 1:i + 2])
            if escaped is None:
                raise _Malformed("malformed_line", line_no, "invalid HELP escape")
            out.append(escaped)
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _parse_exposition(text):
    """Strict Prometheus text format 0.0.4 -> (metadata, series). Raises ``_Malformed``."""
    metadata, series = {}, []
    for line_no, line in enumerate(text[:-1].split("\n"), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            for keyword, field in (("# HELP ", "help"), ("# TYPE ", "type")):
                if not line.startswith(keyword):
                    continue
                name, _, rest = line[len(keyword):].partition(" ")
                if not _NAME.fullmatch(name):
                    raise _Malformed("malformed_line", line_no, f"bad metric name in {field}")
                entry = metadata.setdefault(name, {})
                if field in entry:
                    raise _Malformed("duplicate_metadata", line_no, f"{field} repeated")
                if field == "type" and rest not in _PROM_TYPES:
                    raise _Malformed("malformed_line", line_no, "unknown TYPE")
                entry[field] = rest if field == "type" else _unescape_help(rest, line_no)
            continue  # any other comment is ignored, as in the exposition format
        match = _NAME.match(line)
        if not match:
            raise _Malformed("malformed_line", line_no, "line is not a metric sample")
        rest, labels = line[match.end():], {}
        if rest.startswith("{"):
            labels, rest = _parse_labels(rest[1:], line_no)
        tokens = rest.split()
        if not rest[:1].isspace() or len(tokens) not in (1, 2) or (
                len(tokens) == 2 and not _TIMESTAMP.fullmatch(tokens[1])):
            raise _Malformed("malformed_line", line_no, "expected value [timestamp]")
        kind, value = _parse_value(tokens[0], line_no)
        series.append({"line": line_no, "name": match.group(), "labels": labels,
                       "kind": kind, "value": value, "timestamped": len(tokens) == 2})
    return metadata, series


def _check_device_args(device_key, device_map, clock_id):
    _require(device_key in DEVICE_KEYS,
             f"device_key must be one of {DEVICE_KEYS}; no default is chosen")
    _require(type(device_map) is dict and bool(device_map), "device_map must be a nonempty dict")
    _require(all(_text(k) and _text(v) for k, v in device_map.items()),
             "device_map keys and values must be nonempty text")
    _require(len(set(device_map.values())) == len(device_map),
             "device_map values (allocation devices) must be distinct")
    _require(_text(clock_id), "clock_id must be nonempty text")


def parse_dcgm_prom(text, *, scrape_t_us, device_key, device_map, clock_id, scrape_ref,
                    scope):
    """One dcgm-exporter scrape -> ``gate.s2_telemetry`` samples (L1; format L2).

    ``text`` is the response body (``None`` when the request failed); ``scrape_t_us`` the
    monotonic time read immediately before the request on clock ``clock_id``;
    ``device_key`` the exporter label (``UUID`` or ``gpu``) whose values ``device_map``
    maps to allocation device IDs (``UUID`` is stable across reboots; a ``gpu`` index is
    not; the caller chooses, no default). A device label on a pinned series that is
    absent from ``device_map`` is an ``unmapped_device`` error and that series is not
    emitted (unpinned series are only counted in ``unpinned_metrics``). Every pinned DCGM metric the scrape exposes yields one
    sample per mapped device (``ok``; ``no_data`` when the device's line is missing or
    ``NaN``; ``scrape_failed`` when the scrape or metric is untrustworthy, ``+-Inf``, a
    negative value or a duplicated series). A pinned metric the scrape does not expose
    yields no sample (``pinned_absent``): absence is never zero. A failed, empty,
    sample-less (no ``DCGM_`` line), truncated or malformed scrape makes every pinned
    series ``scrape_failed``.
    """
    _check_scope(scope)
    _check_device_args(device_key, device_map, clock_id)
    _require(_int(scrape_t_us) and scrape_t_us >= 0,
             "scrape_t_us must be a non-negative integer")
    _require(_text(scrape_ref), "scrape_ref must be nonempty text")
    _require(text is None or type(text) is str, "text must be str or None")
    devices = sorted(device_map.values())
    issues, samples, absent, unpinned, unmapped = [], [], [], [], set()
    failure = None
    if text is None:
        failure = ("scrape_missing", "No response body was captured; the scrape failed")
    elif text == "":
        failure = ("scrape_empty", "The scrape body is empty")
    elif not text.endswith("\n"):
        failure = ("scrape_truncated", "The scrape body does not end with a newline")
    else:
        try:
            metadata, series = _parse_exposition(text)
        except _Malformed as error:
            failure = (error.code, f"line {error.line}: {error.message}")
        else:
            if not any(row["name"].startswith("DCGM_") for row in series):
                failure = ("scrape_no_samples", "The scrape body has no DCGM_ sample line")

    def emit(metric, device, status, value, ref):
        samples.append({"metric": metric, "device": device, "clock_id": clock_id,
                        "t_us": scrape_t_us, "status": status, "value": value,
                        "raw_ref": ref})

    if failure is not None:
        issues.append(_issue(failure[0], "scrape", failure[1]))
        for metric in _DCGM_ORDER:
            for device in devices:
                emit(metric, device, "scrape_failed", None, scrape_ref)
    else:
        for metric in _DCGM_ORDER:
            _pinned_metric(metric, metadata, series, device_key, device_map, devices,
                           scrape_ref, emit, issues, absent, unmapped)
        counts = {}
        for row in series:
            if row["name"] not in DCGM_PINNED:
                counts[row["name"]] = counts.get(row["name"], 0) + 1
        unpinned = [{"metric": name, "type": metadata.get(name, {}).get("type"),
                     "series": counts[name]} for name in sorted(counts)]
    for label in sorted(unmapped):
        issues.append(_issue("unmapped_device", f"scrape/device/{label}",
                             "Exporter device is not in device_map; its rows are not "
                             "emitted"))
    status_counts = {}
    for row in samples:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "artifact_kind": SCRAPE_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {
            "text_sha256": None if text is None
            else hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(),
            "device_map": identity_hash(device_map),
        },
        "units": {"t_us": "integer microseconds on the declared monotonic clock",
                  "metrics": {metric: PINNED_METRICS[metric][0] for metric in _DCGM_ORDER}},
        "status": _status(issues, any(item["code"] == "help_not_pinned" for item in issues)),
        "status_meaning": "parse consistency only; never host qualification",
        "clock_id": clock_id,
        "device_key": device_key,
        "scrape": {"t_us": scrape_t_us, "ref": scrape_ref,
                   "outcome": "failed" if failure else "parsed",
                   "failure": failure[0] if failure else None},
        "samples": samples,
        "status_counts": dict(sorted(status_counts.items())),
        "pinned_absent": absent,
        "unpinned_metrics": unpinned,
        "unmapped_devices": sorted(unmapped),
        "host_qualification": "qualify_on_host",
        "issues": _sorted_issues(issues),
    }


def _pinned_metric(metric, metadata, series, device_key, device_map, devices, ref, emit,
                   issues, absent, unmapped):
    pin = DCGM_PINNED[metric]
    rows = [row for row in series if row["name"] == metric]
    path = f"scrape/{metric}"
    if not rows:
        absent.append(metric)
        issues.append(_issue("pinned_metric_absent", path, "Pinned metric not exposed by "
                             "this scrape; no sample is emitted (never zero)", "warning"))
        return
    info = metadata.get(metric, {})
    trusted = True
    if "help" not in info or "type" not in info:
        issues.append(_issue("pinned_metadata_missing", path, "HELP or TYPE missing"))
        trusted = False
    elif info["type"] != pin["type"]:
        issues.append(_issue("pinned_type_mismatch", path,
                             f"TYPE {info['type']!r}, pinned {pin['type']!r}"))
        trusted = False
    elif pin["help"] is None:
        issues.append(_issue("help_not_pinned", path, "No recorded HELP text pins this "
                             "metric's unit", "warning"))
    elif info["help"] != pin["help"]:
        issues.append(_issue("pinned_help_mismatch", path, "HELP differs from the recorded "
                             "exporter; the unit cannot be confirmed"))
        trusted = False
    per_device = {device: [] for device in devices}
    for row in rows:
        label = row["labels"].get(device_key)
        if label is None:
            issues.append(_issue("device_label_missing", f"scrape/line/{row['line']}",
                                 f"Pinned series lacks the {device_key!r} label"))
            trusted = False
        elif label in device_map:
            per_device[device_map[label]].append(row)
        else:
            unmapped.add(label)
    for device in devices:
        found = per_device[device]
        line_ref = f"{ref}#line={found[0]['line']}" if len(found) == 1 else ref
        if not trusted:
            emit(metric, device, "scrape_failed", None, line_ref)
            continue
        if not found:
            emit(metric, device, "no_data", None, ref)
            continue
        if len(found) > 1:
            issues.append(_issue("duplicate_series", f"{path}/{device}",
                                 "More than one series for one device"))
            emit(metric, device, "scrape_failed", None, ref)
            continue
        row = found[0]
        line_path = f"scrape/line/{row['line']}"
        if row["timestamped"]:
            issues.append(_issue("exporter_timestamp_ignored", line_path, "The sample time "
                                 "is scrape_t_us on the declared clock", "warning"))
        if row["kind"] == "nan":
            issues.append(_issue("non_finite_value", line_path, "NaN reads as no_data",
                                 "warning"))
            emit(metric, device, "no_data", None, line_ref)
        elif row["kind"] == "inf":
            issues.append(_issue("non_finite_value", line_path, "Infinite value is not a "
                                 "reading"))
            emit(metric, device, "scrape_failed", None, line_ref)
        elif row["value"] < 0:  # power, memory, utilization, XID code and counters
            code = f"negative_{pin['type']}_value"
            issues.append(_issue(code, line_path, f"{pin['type'].capitalize()} below zero is "
                                 "not a physical reading"))
            emit(metric, device, "scrape_failed", None, line_ref)
        else:
            emit(metric, device, "ok", row["value"], line_ref)


def parse_dcgm_scrapes(scrapes, *, device_key, device_map, clock_id, scope):
    """A time-ordered scrape series -> samples, failed scrapes and counter resets (L1).

    ``scrapes`` is a list of ``{t_us, text, ref}`` with strictly increasing ``t_us``. A
    reset is a decrease between consecutive ``ok`` readings of one counter series; it is
    reported here and summarized by ``summarize_telemetry``, never corrected.
    """
    _check_scope(scope)
    _check_device_args(device_key, device_map, clock_id)
    _require(type(scrapes) is list and bool(scrapes), "scrapes must be a nonempty list")
    reports, previous = [], None
    for position, scrape in enumerate(scrapes):
        _require(type(scrape) is dict and set(scrape) == {"t_us", "text", "ref"},
                 f"scrapes[{position}] keys must be exactly ref, t_us, text")
        _require(_int(scrape["t_us"]), f"scrapes[{position}].t_us must be an integer")
        _require(previous is None or scrape["t_us"] > previous,
                 "scrape times must strictly increase")
        previous = scrape["t_us"]
        reports.append(parse_dcgm_prom(scrape["text"], scrape_t_us=scrape["t_us"],
                                       device_key=device_key, device_map=device_map,
                                       clock_id=clock_id, scrape_ref=scrape["ref"],
                                       scope=scope))
    samples = [row for report in reports for row in report["samples"]]
    resets = []
    for metric in _DCGM_ORDER:
        if DCGM_PINNED[metric]["type"] != "counter":
            continue
        for device in sorted(device_map.values()):
            ok = [(row["t_us"], row["value"]) for row in samples if row["metric"] == metric
                  and row["device"] == device and row["status"] == "ok"]
            for (_, before), (t_us, value) in itertools.pairwise(ok):
                if value < before:
                    resets.append({"metric": metric, "device": device, "t_us": t_us,
                                   "previous": before, "value": value})
    issues, absent, unpinned = [], {}, {}
    for position, report in enumerate(reports):
        issues += [dict(item, path=f"scrapes/{position}/{item['path']}")
                   for item in report["issues"]]
        for metric in report["pinned_absent"]:
            absent[metric] = absent.get(metric, 0) + 1
        for row in report["unpinned_metrics"]:
            entry = unpinned.setdefault(row["metric"], {"types": set(), "scrapes": 0})
            entry["scrapes"] += 1
            if row["type"] is not None:
                entry["types"].add(row["type"])
    return {
        "artifact_kind": SCRAPES_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {"scrape_texts_sha256": [r["input_hashes"]["text_sha256"]
                                                 for r in reports],
                         "device_map": identity_hash(device_map)},
        "units": reports[0]["units"],
        "status": _status(issues, any(r["status"] == "unverified" for r in reports)),
        "status_meaning": "parse consistency only; never host qualification",
        "clock_id": clock_id,
        "device_key": device_key,
        "scrapes": [{**r["scrape"], "status": r["status"]} for r in reports],
        "failed_scrapes": [r["scrape"]["t_us"] for r in reports
                           if r["scrape"]["outcome"] == "failed"],
        "samples": samples,
        "counter_resets": resets,
        "pinned_absent_scrapes": dict(sorted(absent.items())),
        "unpinned_metrics": [{"metric": name, "types": sorted(entry["types"]),
                              "scrapes": entry["scrapes"]}
                             for name, entry in sorted(unpinned.items())],
        "unmapped_devices": sorted({d for r in reports for d in r["unmapped_devices"]}),
        "host_qualification": "qualify_on_host",
        "issues": _sorted_issues(issues),
    }


# ------------------------------------------------------------------ useful attempt rows


def _check_report(report):
    _require(type(report) is dict and set(report) == _REPORT_KEYS,
             "ledger_report must be a reconcile_attempts report")
    _require(all(type(report[k]) is list for k in ("issues", "jobs", "attempts")),
             "ledger_report issues, jobs and attempts must be lists")
    for item in report["issues"]:
        _require(type(item) is dict and _text(item.get("code")), "ledger issue needs a code")
    # reconcile_attempts stores a non-list events container as null with an
    # invalid_json_value issue at path "events"; only that report may carry null events.
    nulled = report["events"] is None and any(
        item["code"] == NON_JSON_CODE and item.get("path") == "events"
        for item in report["issues"])
    _require(type(report["events"]) is list or nulled, "ledger_report events must be a list")
    seen = set()
    for attempt in report["attempts"]:
        _require(type(attempt) is dict and set(attempt) == _ATTEMPT_KEYS,
                 "ledger attempt has an unexpected shape")
        _require(_text(attempt["job_id"]) and _text(attempt["attempt_id"]),
                 "ledger attempt needs job_id and attempt_id")
        _require(attempt["state"] in _STATE_STATUS, "ledger attempt state is unknown")
        _require(attempt["attempt_id"] not in seen, "ledger attempt repeated")
        seen.add(attempt["attempt_id"])
        for key in ("started_timestamp_us", "terminal_timestamp_us"):
            _require(attempt[key] is None or _int(attempt[key]), f"{key} must be int or None")
        _require(type(attempt["events"]) is list, "ledger attempt events must be a list")


def _check_unit_record(record, scope):
    _require(type(record) is dict and set(record) == _UNIT_RECORD_KEYS,
             "unit record keys differ from scripts/s2_run_units.py _Run._envelope")
    _require(record["artifact_kind"] == UNIT_RECORD_KIND
             and _int(record["schema_version"]) and record["schema_version"] == UNIT_RECORD_VERSION,
             f"expected {UNIT_RECORD_KIND} schema version {UNIT_RECORD_VERSION}")
    _require(record["scope"] == scope, "unit record scope differs from the declared scope")
    _require(_text(record["job_id"]), "unit record needs a job_id")
    _require(type(record["steps"]) is list and all(type(s) is dict for s in record["steps"]),
             "unit record steps must be a list of objects")
    _require(_int(record["ledger_events"]) and record["ledger_events"] >= 0,
             "unit record ledger_events must be a non-negative integer")


def useful_attempt_rows(ledger_report, unit_records, *, clock_id, scope):
    """Ledger attempts -> ``summarize_telemetry`` rollout rows with useful flags (L1).

    ``rows`` carry exactly the consumer's keys; ``classification`` gives each attempt's
    ``useful``, ``reason``, terminal type/reason and unit-record state. Attempts without
    a ``started`` timestamp cannot form a row and are listed in ``rejected``.
    """
    _check_scope(scope)
    _require(_text(clock_id), "clock_id must be nonempty text")
    _check_report(ledger_report)
    _require(type(unit_records) is list, "unit_records must be a list")
    for record in unit_records:
        _check_unit_record(record, scope)
    report_hash = _hash(ledger_report, "ledger_report")
    record_hashes = sorted(_hash(r, f"unit_records[{i}]") for i, r in enumerate(unit_records))
    rederived = reconcile_attempts(ledger_report["schedule"], ledger_report["events"])
    non_json = any(item["code"] == NON_JSON_CODE for item in ledger_report["issues"])
    if non_json:
        # The stored values are the nulled ones; compare without what nulling changed.
        matches = (_hash(_without_non_json(rederived), "re-derived ledger report")
                   == _hash(_without_non_json(ledger_report), "ledger_report"))
    else:
        matches = _hash(rederived, "re-derived ledger report") == report_hash
    _require(matches, "ledger_report differs from reconcile_attempts(schedule, events) over "
             "its own inputs; its attempts or issues are not the ledger's")
    issues = []
    if non_json:
        issues.append(_issue("ledger_input_non_json", "ledger_report/issues", "The ledger "
                             "input held non-JSON values stored as null; its evidence cannot "
                             "be re-derived exactly and every useful flag is None"))
    integrity = sorted({item["code"] for item in ledger_report["issues"]
                        if item["code"] not in OUTCOME_ISSUE_CODES})
    if integrity:
        issues.append(_issue("ledger_integrity_issue", "ledger_report/issues",
                             "Ledger evidence is impugned (" + ", ".join(integrity)
                             + "); every useful flag is None"))
    attempts_per_job, events_per_job = {}, {}
    for attempt in ledger_report["attempts"]:
        attempts_per_job[attempt["job_id"]] = attempts_per_job.get(attempt["job_id"], 0) + 1
    for event in ledger_report["events"] or []:
        if type(event) is dict and _text(event.get("job_id")) and _text(event.get("event_id")):
            events_per_job.setdefault(event["job_id"], set()).add(event["event_id"])
    records = {}
    for record in unit_records:
        records.setdefault(record["job_id"], []).append(record)
    for job_id in sorted(records):
        if len(records[job_id]) > 1:
            issues.append(_issue("unit_record_duplicated", f"unit_records/{job_id}",
                                 "More than one unit record for one job"))
        if job_id not in attempts_per_job and any(r["ledger_events"] for r in records[job_id]):
            issues.append(_issue("unit_record_without_ledger_attempts", f"unit_records/{job_id}",
                                 "Unit record counts ledger events the ledger lacks"))

    rows, classification, rejected = [], {}, []
    for attempt in ledger_report["attempts"]:
        attempt_id, job_id, state = attempt["attempt_id"], attempt["job_id"], attempt["state"]
        terminal = None
        if state != "incomplete":
            closing = [e for e in attempt["events"] if type(e) is dict and e.get("type") == state]
            terminal = closing[-1] if closing else None
        mine = records.get(job_id, [])
        unit_state = "missing" if not mine else "duplicated" if len(mine) > 1 else "present"
        useful, reason = _classify(attempt, terminal, integrity, attempts_per_job, mine,
                                   events_per_job, issues)
        classification[attempt_id] = {
            "useful": useful, "reason": reason,
            "terminal_type": terminal["type"] if terminal else None,
            "terminal_reason": terminal.get("reason") if terminal else None,
            "unit_record": unit_state,
        }
        if attempt["started_timestamp_us"] is None:
            issues.append(_issue("attempt_without_start", f"attempts/{attempt_id}",
                                 "Attempt has no started timestamp; no row is formed"))
            classification[attempt_id].update(useful=None, reason="attempt_without_start")
            rejected.append(attempt_id)
            continue
        rows.append({"attempt_id": attempt_id, "job_id": job_id, "clock_id": clock_id,
                     "start_us": attempt["started_timestamp_us"],
                     "end_us": attempt["terminal_timestamp_us"]
                     if state != "incomplete" else None,
                     "status": _STATE_STATUS[state],
                     "useful": classification[attempt_id]["useful"]})
    flags = [entry["useful"] for entry in classification.values()]
    return {
        "artifact_kind": ROWS_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {"ledger_report": report_hash, "unit_records": record_hashes},
        "units": {"start_us": "monotonic microseconds (ledger timestamp_us)",
                  "end_us": "monotonic microseconds (ledger timestamp_us)"},
        "status": _status(issues, None in flags),
        "status_meaning": "attempt accounting consistency only; never scientific admission",
        "clock_id": clock_id,
        "definition": USEFUL_DEFINITION,
        "rows": rows,
        "classification": classification,
        "rejected": rejected,
        "counts": {"attempts": len(flags), "useful_true": flags.count(True),
                   "useful_false": flags.count(False),
                   "useful_unknown": sum(flag is None for flag in flags),
                   "rejected": len(rejected)},
        "issues": _sorted_issues(issues),
    }


def _without_non_json(report):
    """A report minus ``invalid_json_value`` issues and the admission they caused."""
    return {**report,
            "admitted": False,
            "issues": [item for item in report["issues"] if item["code"] != NON_JSON_CODE],
            "jobs": [{**job, "selected_attempt_id": None} if type(job) is dict else job
                     for job in report["jobs"]]}


def _classify(attempt, terminal, integrity, attempts_per_job, mine, events_per_job, issues):
    attempt_id, job_id, state = attempt["attempt_id"], attempt["job_id"], attempt["state"]
    if integrity:
        return None, "ledger_integrity_issue"
    if state == "failed":
        return False, "attempt_failed"
    if state == "cancelled":
        return False, "attempt_cancelled"
    if state == "incomplete":
        return None, "attempt_incomplete"
    if not terminal_established(terminal):
        return None, "completion_not_established"
    if attempts_per_job[job_id] > 1:
        return None, "retry_selection_unregistered"
    path = f"attempts/{attempt_id}"
    if not mine:
        issues.append(_issue("unit_record_missing", path, "No unit record for this job; "
                             "the completion is not cross-checked", "warning"))
        return None, "unit_record_missing"
    if len(mine) > 1:
        return None, "unit_record_duplicated"
    record = mine[0]
    listed = [step for step in record["steps"] if step.get("id") == "ledger_terminal"]
    entries = listed[0].get("attempts") if len(listed) == 1 else None
    expected = {"attempt_id": attempt_id, "type": "completed", "reason": terminal["reason"]}
    if type(entries) is not list or expected not in entries:
        issues.append(_issue("unit_record_disagrees", path, "Unit record's ledger_terminal "
                             "step does not list this completion"))
        return None, "unit_record_disagrees"
    if record["ledger_events"] != len(events_per_job.get(job_id, ())):
        issues.append(_issue("unit_record_event_count_mismatch", path, "Unit record and "
                             "ledger disagree on the job's event count"))
        return None, "unit_record_event_count_mismatch"
    return True, "established_completion"


def _hash(value, what):
    """Canonical-JSON SHA-256 of an input; an input that has none fails closed."""
    try:
        return identity_hash(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{what} is not canonical JSON: {error}") from None


# ------------------------------------------------------------------ allocation partition


def _utc_us(text, what):
    _require(type(text) is str, f"{what} must be ISO 8601 text")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{what} is not ISO 8601") from None
    _require(moment.tzinfo is not None and moment.utcoffset() == timedelta(0),
             f"{what} must carry a zero UTC offset")
    return (moment - _EPOCH) // _US


def _check_billed(billed):
    _require(type(billed) is dict and set(billed) == _BILLED_KEYS,
             "billed_interval keys must be exactly " + ", ".join(sorted(_BILLED_KEYS)))
    for key in ("allocation_id", "clock_id", "source_ref"):
        _require(_text(billed[key]), f"billed_interval.{key} must be nonempty text")
    count, devices = billed["gpu_count"], billed["devices"]
    _require(_int(count) and count >= 1, "gpu_count must be a positive integer")
    _require(type(devices) is list and len(devices) == count
             and all(_text(d) for d in devices) and len(set(devices)) == count,
             "devices must list gpu_count distinct device IDs")
    start, end = billed["start_us"], billed["end_us"]
    _require(_int(start) and _int(end) and 0 <= start < end,
             "billed start_us < end_us must be non-negative integers")


def _check_anchor(anchor, clock_id):
    _require(type(anchor) is dict and set(anchor) == _ANCHOR_KEYS,
             "clock_anchor keys must be exactly " + ", ".join(sorted(_ANCHOR_KEYS)))
    _require(anchor["clock_id"] == clock_id, "clock_anchor names another clock")
    _require(_int(anchor["monotonic_us"]), "clock_anchor.monotonic_us must be an integer")
    _require(_int(anchor["uncertainty_us"]) and anchor["uncertainty_us"] >= 0,
             "clock_anchor.uncertainty_us must be a non-negative integer")
    _require(_text(anchor["anchor_ref"]), "clock_anchor.anchor_ref must be nonempty text")
    return anchor["monotonic_us"] - _utc_us(anchor["utc"], "clock_anchor.utc")


def _verify_chain(records):
    _require(type(records) is list, "lifecycle_journal_records must be a list")
    previous = _GENESIS
    for index, item in enumerate(records):
        _require(type(item) is dict and set(item) == _JOURNAL_KEYS,
                 "lifecycle record shape differs from gate.s2_execution.LifecycleJournal")
        body = {"seq": item["seq"], "prev_hash": item["prev_hash"], "payload": item["payload"]}
        _require(_int(item["seq"]) and item["seq"] == index and item["prev_hash"] == previous
                 and item["record_hash"] == identity_hash(body),
                 "lifecycle journal hash chain is broken")
        previous = item["record_hash"]


def _check_runner(records):
    _require(type(records) is list, "runner_journal_records must be a list")
    events, seen = [], {}
    for position, event in enumerate(records):
        _require(_valid_event(event), f"runner_journal_records[{position}] is not a valid "
                 "gate.s2_ledger event")
        fingerprint = identity_hash(event)
        if event["event_id"] in seen:
            _require(seen[event["event_id"]] == fingerprint,
                     "runner event ID repeated with different content")
            continue  # identical repeat: idempotent, as in reconcile_attempts
        seen[event["event_id"]] = fingerprint
        events.append(event)
    return events


def _check_rows(rows, clock_id):
    _require(type(rows) is list, "attempt_rows must be a list")
    by_attempt = {}
    for row in rows:
        _require(type(row) is dict and set(row) == _ROW_KEYS, "attempt row has an unexpected "
                 "shape (use useful_attempt_rows(...)['rows'])")
        _require(row["clock_id"] == clock_id, "attempt row uses another clock")
        _require(_text(row["attempt_id"]) and _text(row["job_id"]), "attempt row needs ids")
        _require(row["status"] in ROLLOUT_STATUSES, "attempt row status is unknown")
        _require(_int(row["start_us"]), "attempt row start_us must be an integer")
        _require((row["end_us"] is None) == (row["status"] == "unresolved")
                 and (row["end_us"] is None or _int(row["end_us"])),
                 "attempt row end_us must be an integer, or None exactly when unresolved")
        _require(row["useful"] in (True, False, None) and (
            row["useful"] is None or type(row["useful"]) is bool), "useful must be a bool or None")
        _require(row["attempt_id"] not in by_attempt, "attempt row repeated")
        by_attempt[row["attempt_id"]] = row
    return by_attempt


def _bind_row(row, terminal):
    """A row's status and useful flag must follow from the journal's terminal event."""
    kind = "unresolved" if terminal is None else terminal["type"]
    _require(row["status"] == kind, f"attempt row {row['attempt_id']} status "
             f"{row['status']!r} contradicts the runner journal terminal {kind!r}")
    if row["useful"] is True:
        _require(kind == "completed" and terminal_established(terminal),
                 f"attempt row {row['attempt_id']} is useful without an established "
                 "completion in the runner journal")
    elif row["useful"] is False:
        _require(kind in ("failed", "cancelled"), f"attempt row {row['attempt_id']} is "
                 "not useful although the runner journal does not record it failed or "
                 "cancelled")


def _runner_units(events, rows, billed_end, windows, issues):
    """Per-job unit intervals and labels from ledger events and attempt rows."""
    jobs, regression_overlays, last = {}, [], None
    for index, event in enumerate(events):
        t_us = event["timestamp_us"]
        if last is not None and t_us < last:
            issues.append(_issue("runner_timestamp_regression", f"runner/{index}",
                                 "Runner timestamps regress in recorded order"))
            regression_overlays.append((t_us, last, "runner_timestamp_regression"))
        last = t_us if last is None else max(last, t_us)
        job = jobs.setdefault(event["job_id"], {"first": t_us, "last": t_us, "index": index,
                                                "attempts": {}})
        job["first"], job["last"], job["index"] = min(job["first"], t_us), max(
            job["last"], t_us), index
        attempt = job["attempts"].setdefault(event["attempt_id"],
                                             {"start": None, "end": None, "terminal": None})
        slot = "start" if event["type"] == "started" else "end"
        if attempt[slot] is not None:
            issues.append(_issue("runner_attempt_inconsistent", f"runner/{index}",
                                 "Attempt started or ended twice"))
        else:
            attempt[slot] = t_us
            if slot == "end":
                attempt["terminal"] = event
    event_attempts = {aid for job in jobs.values() for aid in job["attempts"]}
    _require(set(rows) <= event_attempts, "attempt rows name attempts the runner journal lacks")
    units = []
    for job_id in sorted(jobs, key=lambda j: (jobs[j]["first"], j)):
        job, flags, unresolved, missing = jobs[job_id], [], False, False
        for attempt_id, attempt in sorted(job["attempts"].items()):
            row = rows.get(attempt_id)
            if attempt["start"] is None:
                _require(row is None, "attempt row names an attempt the runner never started")
                missing = True
                issues.append(_issue("attempt_without_start", f"runner/{attempt_id}",
                                     "Runner attempt has no started event"))
                continue
            _require(row is not None and row["job_id"] == job_id
                     and row["start_us"] == attempt["start"] and row["end_us"] == attempt["end"],
                     "attempt rows do not match the runner journal")
            _bind_row(row, attempt["terminal"])
            unresolved = unresolved or attempt["end"] is None
            flags.append(row["useful"])
        end = job["last"]
        if unresolved:
            later = [e["timestamp_us"] for e in events[job["index"] + 1:]
                     if e["timestamp_us"] >= job["last"]]
            window_end = next((b for a, b in windows or () if a <= job["first"] < b), None)
            end = later[0] if later else window_end if window_end is not None else billed_end
            label = ("unknown", "runner", "attempt_unresolved", job_id)
        elif missing or None in flags or not flags or len(set(flags)) > 1:
            label = ("unknown", "runner", "unit_classification_unresolved", job_id)
        elif flags[0] is True:
            label = ("useful_rollout", "runner", None, job_id)
        else:
            label = ("retry_or_failed", "runner", None, job_id)
        units.append({"job_id": job_id, "start": job["first"], "end": end, "label": label,
                      "events": [e["timestamp_us"] for e in events if e["job_id"] == job_id]})
    return units, regression_overlays


def allocation_from_journals(lifecycle_journal_records, runner_journal_records, *,
                             billed_interval, clock_anchor, attempt_rows, scope):
    """Billed time -> a nonoverlapping, exactly summing phase partition (L1).

    ``lifecycle_journal_records`` are ``LifecycleJournal.read`` records (chain verified
    here); ``runner_journal_records`` the runner's ledger events; ``attempt_rows`` the
    ``rows`` of ``useful_attempt_rows`` for the same events; ``billed_interval`` the
    billed allocation on the runner's monotonic clock; ``clock_anchor`` maps lifecycle
    UTC onto that clock (``None`` -> ``blocked: clock_anchor_missing``, lifecycle time
    unknown, runner units kept). ``telemetry_allocation`` is the ``summarize_telemetry``
    allocation (known segments only, ``PHASE_MAP`` names) or ``None`` when nothing is
    known. Malformed, tampered or mutually inconsistent inputs raise ``ValueError``.
    """
    _check_scope(scope)
    _check_billed(billed_interval)
    clock_id = billed_interval["clock_id"]
    _verify_chain(lifecycle_journal_records)
    offset = None if clock_anchor is None else _check_anchor(clock_anchor, clock_id)
    events = _check_runner(runner_journal_records)
    rows = _check_rows(attempt_rows, clock_id)
    start, end = billed_interval["start_us"], billed_interval["end_us"]
    uncertainty = 0 if clock_anchor is None else clock_anchor["uncertainty_us"]
    issues, overlays, changes = [], [], []
    for item in lifecycle_journal_records:
        payload = item["payload"]
        if type(payload) is dict and payload.get("type") == "transition":
            state = payload.get("state")
            _require(type(state) is dict and state.get("state") in _LIFECYCLE,
                     f"lifecycle record {item['seq']} has an unknown state")
    if offset is None:
        issues.append(_issue("clock_anchor_missing", "clock_anchor", "No anchor maps lifecycle "
                             "UTC onto the runner clock; lifecycle time is unknown"))
        default = ("unknown", "lifecycle", "clock_anchor_missing", None)
        windows = None
    else:
        default = ("unknown", "lifecycle", "before_first_lifecycle_record", None)
        last = None
        for item in lifecycle_journal_records:
            payload = item["payload"]
            if type(payload) is not dict or payload.get("type") != "transition":
                continue
            t_us = _utc_us(item["at_utc"], f"lifecycle record {item['seq']} at_utc") + offset
            if last is not None and t_us < last:
                issues.append(_issue("lifecycle_time_regression", f"lifecycle/{item['seq']}",
                                     "Lifecycle wall time regresses; the span is unknown"))
                overlays.append((t_us, last, "lifecycle_time_regression"))
                t_us = last
            last = t_us
            phase, reason = _LIFECYCLE[payload["state"]["state"]]
            label = (phase, "lifecycle", reason, None)
            if changes and changes[-1][0] == t_us:
                changes.pop()
            if (changes[-1][1] if changes else default) != label:
                changes.append((t_us, label))
        if uncertainty:
            overlays += [(t - uncertainty, t + uncertainty, "clock_alignment_uncertainty")
                         for t, _ in changes]
        windows = [(t, changes[i + 1][0] if i + 1 < len(changes) else math.inf)
                   for i, (t, label) in enumerate(changes) if label[0] == "running"]
    units, regressions = _runner_units(events, rows, end, windows, issues)
    overlays += regressions
    if windows is not None:
        for unit in units:
            if any(not any(a - uncertainty <= t <= b + uncertainty for a, b in windows)
                   for t in unit["events"]):
                issues.append(_issue("runner_outside_running_phase", f"runner/{unit['job_id']}",
                                     "Runner events lie outside the lifecycle running phase"))
        for a, b in windows:
            if not any(u["start"] < b and u["end"] > a for u in units):
                issues.append(_issue("running_without_runner_records", "runner",
                                     "Running phase has no runner records; it is unknown",
                                     "warning"))
    ordered = sorted(units, key=lambda u: (u["start"], u["end"]))
    if any(b["start"] < a["end"] for a, b in itertools.pairwise(ordered)):
        issues.append(_issue("runner_units_overlap", "runner", "Runner units overlap"))
    change_times = [t for t, _ in changes]

    def lifecycle_at(x):
        position = bisect.bisect_right(change_times, x)
        return changes[position - 1][1] if position else default

    def label_at(x):
        hits = sorted(reason for a, b, reason in overlays if a <= x < b)
        if hits:
            return ("unknown", "overlay", hits[0], None)
        covering = [u for u in units if u["start"] <= x < u["end"]]
        window = next(((a, b) for a, b in windows or () if a <= x < b), None)
        if len(covering) > 1:
            return ("unknown", "runner", "runner_units_overlap", None)
        if covering:
            if windows is not None and window is None:
                return ("unknown", "runner", "runner_outside_running_phase",
                        covering[0]["job_id"])
            return covering[0]["label"]
        life = lifecycle_at(x)
        if life[0] != "running":
            return life
        inside = [u["start"] for u in units if u["start"] < window[1] and u["end"] > window[0]]
        if not inside:
            return ("unknown", "runner", "running_without_runner_records", None)
        return ("unit_startup" if x < min(inside) else "idle", "runner", None, None)

    points = {start, end, *change_times}
    for a, b, _ in overlays:
        points.update((a, b))
    for unit in units:
        points.update((unit["start"], unit["end"]))
    for a, b in windows or ():
        points.update((a, b))
    points = sorted(p for p in points if start <= p <= end)
    segments = []
    for x, y in itertools.pairwise(points):
        label = label_at(x)
        if segments and segments[-1]["label"] == label:
            segments[-1]["end_us"] = y
        else:
            segments.append({"label": label, "start_us": x, "end_us": y})
    segments = [{"phase": s["label"][0], "start_us": s["start_us"], "end_us": s["end_us"],
                 "source": s["label"][1], "reason": s["label"][2], "job_id": s["label"][3]}
                for s in segments]
    totals = dict.fromkeys(SEGMENT_PHASES, 0)
    for segment in segments:
        totals[segment["phase"]] += segment["end_us"] - segment["start_us"]
    if sum(totals.values()) != end - start:  # construction invariant; never expected
        raise RuntimeError("partition does not sum to billed time")
    unknown = []
    for segment in segments:
        if segment["phase"] != "unknown":
            continue
        if unknown and unknown[-1][1] == segment["start_us"]:
            unknown[-1][1] = segment["end_us"]
        else:
            unknown.append([segment["start_us"], segment["end_us"]])
    if unknown:
        reasons = {}
        for segment in segments:
            if segment["phase"] == "unknown":
                reasons[segment["reason"]] = reasons.get(segment["reason"], 0) + (
                    segment["end_us"] - segment["start_us"])
        issues.append(_issue("unknown_time", "segments", "Billed time not attributable: "
                             + ", ".join(f"{k}={v} us" for k, v in sorted(reasons.items())),
                             "warning"))
    known = [{"phase": PHASE_MAP[s["phase"]], "start_us": s["start_us"], "end_us": s["end_us"]}
             for s in segments if s["phase"] != "unknown"]
    telemetry_allocation = None if not known else {
        "allocation_id": billed_interval["allocation_id"], "clock_id": clock_id,
        "gpu_count": billed_interval["gpu_count"], "devices": list(billed_interval["devices"]),
        "billed_start_us": start, "billed_end_us": end, "segments": known}
    return {
        "artifact_kind": PARTITION_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {"lifecycle_records": _hash(lifecycle_journal_records,
                                                    "lifecycle_journal_records"),
                         "runner_records": _hash(runner_journal_records,
                                                 "runner_journal_records"),
                         "attempt_rows": _hash(attempt_rows, "attempt_rows"),
                         "billed_interval": _hash(billed_interval, "billed_interval"),
                         "clock_anchor": None if clock_anchor is None
                         else _hash(clock_anchor, "clock_anchor")},
        "units": {"time": "integer microseconds on the billed_interval monotonic clock"},
        "status": _status(issues, bool(unknown)),
        "status_meaning": "partition consistency only; never host qualification",
        "clock_id": clock_id,
        "billed_interval": dict(billed_interval),
        "clock_anchor": None if clock_anchor is None else dict(clock_anchor),
        "phase_map": dict(PHASE_MAP),
        "segments": segments,
        "totals_us": totals,
        "unknown_intervals_us": unknown,
        "telemetry_allocation": telemetry_allocation,
        "host_qualification": "qualify_on_host",
        "issues": _sorted_issues(issues),
    }
