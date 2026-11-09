"""Actual AlpaSim unit outputs -> W3 observations, adapter sidecars and gate records.

Tickets: completeness-audit high gap "S2.3/S2.5/S4.6 actual extraction" (module F2 of the
final D-024 round); build-spec B1 consumers (lane R1 F3 real-capture admission path), A4 C2/C3
(attempt evidence from logs and folders; exit code never success), P1/P2 (attempt list,
``run_uuid`` from ``run_metadata.yaml``), A5 (``describe_validity`` input), A7/F8
(``evaluate_registered`` records). Agent-authored evaluator engineering under D-018/D-024;
serves S2.1-S2.7 [C] (G1, G2, G3-R1, G7) and the S4.6 evaluation input path (G9).

Two pure entry points over receipted bytes (no subprocess, network, clock or writes):

- ``extract_unit(root, receipts, unit_record, ledger_report, *, scope)`` reads one unit's
  RUN_DIR (lane R2 section 2: ``rollouts/<clipgt_id>/<rollout_uuid>/{rollout.asl,
  metrics.parquet, _complete}``, ``aggregate/results-summary.json``, ``run_metadata.yaml``,
  ``physloop-capture/compose-{up.stdout,logs}.txt``) only through receipts
  (``gate.s2_identity.read_artifact``: no-follow, hash-checked). It re-derives the runner's
  attempt evidence with ``gate.s2_capture.reconcile_unit_attempts`` from the same bytes and
  binds it three ways, all blocking when they differ: the unit record's ``reconcile``
  ``evidence_ref`` must equal ``unit_attempt_evidence:<identity_hash(report)>``; the ledger
  attempts of the job (``gate.s2_ledger.reconcile_attempts`` output) must equal
  ``gate.s2_executor.unit_terminal_events`` of that report in order, state, reason and
  upstream identity; and each terminal ledger event must cite that evidence reference.
  The runner derives those events with its recorder ``interruption``
  (s2_run_units.py:959-960); it is re-derived from the unit record (``_runner_interruption``)
  and reported as ``attempt_evidence.runner_interruption``.
  ``receipts`` must be the ``files`` of the unit's stored
  ``physloop-capture/run-dir-manifest.json`` (hashed before teardown), not a re-walk of the
  directory: ``identity_hash`` of the path-sorted receipts must equal the unit record's single
  ``run_dir_hash`` step ``files_sha256`` with that step ``passed`` (s2_run_units.py:799-802),
  else ``run_dir_manifest_unbound`` blocks. That binds every receipted byte (metrics.parquet,
  ``_complete``, ``rollout.asl`` included), not only the paths the evidence hash covers; a
  later file (e.g. ``unit-attempt-evidence.json``) also yields ``attempt_evidence_unbound``.
  Read bounds equal the runner's per file class (logs and summary ``MAX_LOG_BYTES``,
  ``run_metadata.yaml`` ``MAX_CONFIG_BYTES``), so both read the same bytes.
  Identities therefore come from the ledger (``scene_id == clipgt_id`` is enforced) and
  ``run_uuid`` from ``run_metadata.yaml`` (``gate.s2_capture.bind_run_uuid``).
- ``assemble_batch(unit_extractions, ledger_report, *, scope, observed_evaluator_sha,
  conventions)`` runs ``gate.alpasim_adapter.normalize_summary`` once per unit (each unit's
  own summary; summaries are never merged), ``gate.s2_diagnostics.describe_validity`` over
  all observations, and emits ``gate_inputs`` (``records`` + ``adapter_metric``) for
  ``gate.s2_gate.evaluate_registered`` only when admitted.

Observation rows carry exactly keys ``describe_validity`` accepts (binding keys, W3 keys and
the ``corridor_departure`` extension); every other fact sits in the unit sidecar
(``attempts[*]``). One row per ledger attempt with an upstream identity, metrics or not;
an attempt without identity (pre-UUID crash) is counted, never given an invented identity.

Pinned-source semantics (AlpaSim e121e591; vendored ``tests/fixtures/alpasim_s2_raw`` and
the R2 lane clone):

- metrics.parquet is the long table of ``create_metrics_dataframe`` (data.py:2027-2077:
  values cast to float64 at :2056, list columns exploded at :2077), written by
  runtime_evaluator.py:53; columns ``gate.s2_extract.PARQUET_COLUMNS``. Its ``run_uuid`` is
  ``run_metadata.yaml``'s (runtime_evaluator.py:193).
- Every pinned required scorer writes ``valid=True`` for every sample (ground_truth.py:156-215,
  collision.py:57, offroad.py:366, image.py:60), so ``valid=False``/null in a required stream
  is an anomaly: it is preserved, counted per stream and makes ``required_metrics_verified``
  false. Upstream aggregation ignores ``valid`` (processing.py:744-748 pivots ``values`` only).
- ``full_max_deviation_m`` is the maximum of every ``dist_to_gt_trajectory`` sample as
  written, ``valid`` flag included (agent technical default, D-018: an invalid-flagged sample
  is still a written deviation, and excluding it could hide a >= 4 m excursion; the invalid
  count is reported regardless). A missing stream, empty stream or a null/nonfinite/negative
  value gives ``None``, never 0. Arrow types must be exactly the writer's: ``values``
  float64, ``timestamps_us`` int64, ``valid`` bool (else ``metrics_schema_differs``).
  Upstream summary aggregates are truncated after the first offroad/collision
  (processing.py:63) and after ``dist_to_gt_trajectory >= 4`` (main.py:155-160),
  the event row itself kept (modifiers.py:99-101), and timesteps before ``eval_relevant``
  are dropped (processing.py:38), so the summary never supplies the full maximum; a summary
  value below the full maximum is reported (``summary_deviation_below_full_stream``, info).
  The summary aggregates a subset of the same samples, so a value above it means parquet
  and summary disagree (``summary_deviation_above_full_stream``, blocking). A summary that
  is not schema_version 1 with a rollouts list blocks (``summary_invalid``).
- Upstream zero-fills a missing ``offroad`` column (processing.py:752-760). Here a missing
  required stream is listed and ``required_metrics_verified`` is false; a summary value whose
  raw stream is absent is reported (``summary_metric_without_raw_stream``). Nothing is filled.
- ``actual_steps`` is the attempt's single logged ``Session STARTING ... steps=`` value
  (event_loop.py:622-627), which is the clipped ``n_sim_steps_actual`` (unbound_rollout.py:
  152-155): the planned horizon, not a count of executed steps. A driver-ended session
  (``EndSimulationException``, event_loop.py:652-653) still writes ``_complete`` and the
  COMPLETED line, with the planned duration (event_loop.py:83-85, 676-695); only
  ``fail_on_driver_termination: true`` turns it into ``DriverRequestedTerminationError``
  (event_loop.py:476-478, 646-651), and the EvalConfig default is false. So both
  ``requested_steps`` and ``actual_steps`` are reported only when the receipted
  ``eval-config.yaml`` (wizard ``_generate_eval_config``, configuration.py:241-249) has
  ``fail_on_driver_termination: true`` (pinned preset e2e_challenge/dev.yaml:51); absent
  (``eval_config_absent``) or not true (``driver_termination_not_fatal``) withholds them
  (warning; the adapter then blocks with ``horizon_unverified``); YAML that cannot be parsed
  or constructed (including nesting past the recursion limit, and a mapping that repeats a
  key, which PyYAML would otherwise resolve silently to its last value) blocks
  (``eval_config_unreadable``).
  Several or no logged values also leave both absent (unknown). ``requested_steps`` comes
  from the ledger job.
- ``completion_validated`` is true only when the re-derived evidence marks the attempt
  ``complete`` (``_complete`` marker, metrics, exactly one non-failed summary row, Session
  STARTING and COMPLETED lines; ``gate.s2_capture``) and the ledger attempt is ``completed``.
  The Compose exit code is never consulted (simulate/__main__.py:231-265).
- Failed summary rows carry ``run_uuid`` null and ``rollout_id`` ``failed-<idx>`` without an
  allocated UUID (simulate/__main__.py:58-60; failed_rollouts.py:44-56); they stay in the
  unit's summary verbatim, so the adapter rejects their identity instead of losing them.
- ``corridor_departure`` is whether any full-stream ``left_corridor_laterally`` sample is
  positive (agent technical default mapping, D-018; whether it invalidates a unit is Huy's
  validity definition). ``black_frame_count`` counts positive ``img_is_black`` samples, a
  single-camera lower bound (image.py:46-62). Render, map and pose coverage are not
  establishable from metrics and stay unknown (absent); ``endpoint_overshoot`` is absent.

Owner conventions (Huy; required keyword, no default). ``conventions=None`` keeps
``admission`` ``blocked`` with code ``owner_conventions_unregistered`` while every fact is
still computed. Exact keys: ``schema_version`` (1), ``conventions_id`` and ``source_ref``
(text), ``terminal_semantics`` and ``validity_definition`` (the ``describe_validity``
schema, passed through unchanged). ``terminal_semantics`` must be
``exact_requested_horizon``; ``accept_clipped_horizon`` is recognised but blocks
(``terminal_semantics_unsupported``) because the unchanged adapter rejects
``actual_steps != requested_steps``. A supplied object is hashed and labelled
``supplied_not_authenticated``. With valid conventions, records are emitted only when the
ledger admitted the batch, every unit extraction passed, every unit normalization admitted,
the validity report is consistent and every job is ``valid`` under the definition (invalid
or unknown blocks; nothing is excluded, because exclusion is Huy's missing-data rule).

Verification level per function (build-spec section 1): ``extract_unit`` and
``assemble_batch`` L1 over hand-written fixtures in the pinned output shapes
(``tests/fixtures/s2_actual_extract``) with literal known answers; L2 for the parquet schema,
summary row shapes, log formats and aggregation semantics (pinned sources cited above); real
outputs are L3 (``qualify_on_host``). A green test is not ``ACTUAL_INPUT_QUALIFIED``.

Agent technical defaults (D-018): read bounds ``gate.s2_actual.MAX_PARQUET_BYTES`` (256 MiB)
for metrics and the runner's ``MAX_LOG_BYTES`` (512 MiB) / ``MAX_CONFIG_BYTES`` (16 MiB) for
the files it also reads (provisional until the first measured unit); the valid-flag rule
for the full deviation maximum; the corridor/black-frame mappings above; the
``fail_on_driver_termination`` horizon attestation. ``observed_evaluator_sha`` is a caller
argument labelled ``caller_supplied_not_authenticated`` in the batch artifact (no checkout
record is an input here).

This module never executes a process, writes files, follows symlinks, reads outside
receipted files, fills a missing sample or metric, selects among retried attempts, infers
success from an exit code, edits or copies ``gate/verdict.py``/``gate/stats/``, accepts
synthetic inputs or diagnostic units (``s2_diagnostic_unit_record``, a ``role`` or
``inference_input`` key, or a ``diagnostic_qualification`` ledger job: review G2-04), or
chooses an owner value.
"""

import hashlib
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gate.alpasim_adapter import IDENTITY_FIELDS, SUPPORTED_EVALUATOR_SHA, normalize_summary
from gate.s2_actual import (
    LOG_PATHS,
    MAX_PARQUET_BYTES,
    RUN_METADATA_PATH,
    SUMMARY_PATH,
    inspect_metrics_parquet,
)
from gate.s2_capture import bind_run_uuid, reconcile_unit_attempts
from gate.s2_diagnostics import ENVELOPE_M, describe_validity
from gate.s2_executor import unit_terminal_events
from gate.s2_extract import RAW_AGGREGATIONS
from gate.s2_identity import identity_hash, read_artifact, read_json_bytes

# Equal to scripts/s2_run_units.py MAX_LOG_BYTES / MAX_CONFIG_BYTES (test-pinned; gate/
# never imports scripts/): a file between two different bounds re-derives other evidence.
MAX_LOG_BYTES = 512 * 1024 * 1024  # agent technical default (D-018)
MAX_CONFIG_BYTES = 16 * 1024 * 1024  # agent technical default (D-018)
EVAL_CONFIG_PATH = "eval-config.yaml"
# Everything the SafeLoader-based load can raise on hostile bytes: parser/scanner/constructor errors
# (YAMLError), scalar values a constructor rejects (ValueError, TypeError, OverflowError)
# and nesting deeper than the interpreter recursion limit (RecursionError).
_YAML_LOAD_ERRORS = (yaml.YAMLError, ValueError, TypeError, OverflowError, RecursionError)
_MERGE_TAG = "tag:yaml.org,2002:merge"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping repeating a key (agent technical default (D-018)).

    PyYAML keeps the last value, so ``fail_on_driver_termination: false`` followed by
    ``... true`` would load as true. A ``<<`` merge key is skipped here: keys it merges in may
    be overridden by explicit keys, as YAML defines. An unhashable key raises TypeError."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen = set()
            for key_node, _value_node in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping", node.start_mark,
                        f"found duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
        return super().construct_mapping(node, deep=deep)
UNIT_ARTIFACT_KIND = "s2_actual_unit_extraction"
BATCH_ARTIFACT_KIND = "s2_actual_batch_extraction"
SCHEMA_VERSION = 1
SCOPES = ("public_source_fixture", "actual")
UNIT_RECORD_KIND = "s2_unit_record"
UNIT_RECORD_VERSION = 1
# Declared diagnostic units (review G2-04; gate.s2_packet DIAGNOSTIC_PHASE) are refused by name.
DIAGNOSTIC_UNIT_RECORD_KIND = "s2_diagnostic_unit_record"
DIAGNOSTIC_PHASE = "diagnostic_qualification"
EVIDENCE_REF_PREFIX = "unit_attempt_evidence:"
PROVENANCE_PREFIX = "s2_actual_extract:"
DEVIATION_STREAM = "dist_to_gt_trajectory"
CORRIDOR_STREAM = "left_corridor_laterally"
BLACK_STREAM = "img_is_black"
CONVENTION_FIELDS = frozenset(
    {"schema_version", "conventions_id", "source_ref", "terminal_semantics",
     "validity_definition"})
TERMINAL_SEMANTICS = ("exact_requested_horizon",)
UNSUPPORTED_TERMINAL_SEMANTICS = ("accept_clipped_horizon",)
_LEDGER_KEYS = {"admitted", "issues", "jobs", "attempts", "counts", "schedule", "events"}
_ATTEMPT_KEYS = {"job_id", "attempt_id", "parent_attempt_id", "retry_reason",
                 "upstream_identity", "state", "started_timestamp_us", "terminal_timestamp_us",
                 "terminal_reason", "events"}
_STATE_FOR_EVENT = {"completed": "completed", "failed": "failed", "cancelled": "cancelled"}
# scripts/s2_run_units.py _Run._envelope (614-620) over the unit record (491-494).
_UNIT_RECORD_KEYS = frozenset({
    "artifact_kind", "schema_version", "scope", "producer", "input_hashes", "units", "status",
    "status_reasons", "index", "job_id", "scene_id", "steps", "outcome", "stop_run",
    "compose_exit_code", "unit_outcome", "ledger_events", "issues"})
# gate.s2_ledger._JOB_FIELDS plus the reconciled fields (s2_ledger.py:210).
_LEDGER_JOB_KEYS = frozenset({
    "job_id", "phase", "arm", "policy_id", "scene_id", "road_id", "stratum", "replicate_id",
    "protocol_hash", "requested_steps", "attempt_ids", "status", "selected_attempt_id"})
_PINNED_COLUMN_TYPES = {"values": str(pa.float64()), "timestamps_us": str(pa.int64()),
                        "valid": str(pa.bool_())}
RUN_DIR_HASH_STEP = "run_dir_hash"


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _text(value):
    return type(value) is str and bool(value.strip())


def _issue(issues, code, path, message, severity="blocking", refs=()):
    issues.append({"code": code, "path": path, "severity": severity,
                   "evidence_refs": sorted(set(refs)), "message": message})


def _blocking(issues):
    return any(item["severity"] == "blocking" for item in issues)


def _check_scope(scope):
    if scope == "synthetic":
        raise ValueError("synthetic inputs are refused; scope must be public_source_fixture "
                         "or actual")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; no default is chosen")


def _check_ledger(ledger_report):
    if (type(ledger_report) is not dict or set(ledger_report) != _LEDGER_KEYS
            or type(ledger_report["jobs"]) is not list
            or type(ledger_report["attempts"]) is not list
            or type(ledger_report["admitted"]) is not bool):
        raise ValueError("ledger_report must be the unmodified reconcile_attempts output")
    for job in ledger_report["jobs"]:
        if type(job) is not dict or not _text(job.get("job_id")):
            raise ValueError("ledger jobs must carry text job IDs")
        if job.get("phase") == "synthetic":
            raise ValueError("synthetic inputs are refused; ledger job "
                             f"{job['job_id']!r} has phase synthetic")
        if job.get("phase") == DIAGNOSTIC_PHASE:
            raise ValueError("diagnostic units are refused; ledger job "
                             f"{job['job_id']!r} has phase {DIAGNOSTIC_PHASE} and is never an "
                             "inference observation")
        if (set(job) != _LEDGER_JOB_KEYS
                or not all(_text(job[key]) for key in ("policy_id", "scene_id", "road_id",
                                                       "stratum", "arm", "protocol_hash"))
                or type(job["replicate_id"]) is not int
                or type(job["requested_steps"]) is not int or job["requested_steps"] < 1
                or type(job["attempt_ids"]) is not list):
            raise ValueError("ledger jobs must be unmodified reconcile_attempts jobs with "
                             f"exactly {sorted(_LEDGER_JOB_KEYS)}")
    for attempt in ledger_report["attempts"]:
        if (type(attempt) is not dict or set(attempt) != _ATTEMPT_KEYS
                or type(attempt["events"]) is not list
                or (attempt["upstream_identity"] is not None
                    and (type(attempt["upstream_identity"]) is not dict
                         or set(attempt["upstream_identity"]) != set(IDENTITY_FIELDS)))):
            raise ValueError("ledger attempts must be unmodified reconcile_attempts attempts")


def _check_receipts(receipts):
    if type(receipts) is not list or not all(
            type(item) is dict and set(item) == {"path", "sha256", "size_bytes"}
            and type(item["path"]) is str and type(item["size_bytes"]) is int
            for item in receipts):
        raise ValueError("receipts must be a list of exactly {path, sha256, size_bytes}")
    paths = [item["path"] for item in receipts]
    if len(set(paths)) != len(paths):
        raise ValueError("receipt paths must be unique")


def _check_unit_record(unit_record, scope):
    if type(unit_record) is dict and (
            unit_record.get("artifact_kind") == DIAGNOSTIC_UNIT_RECORD_KIND
            or "role" in unit_record or "inference_input" in unit_record):
        raise ValueError("diagnostic units are refused: a unit record with artifact_kind "
                         f"{DIAGNOSTIC_UNIT_RECORD_KIND!r} or a role/inference_input key is a "
                         "diagnostic unit, never an inference observation")
    if (type(unit_record) is not dict
            or unit_record.get("artifact_kind") != UNIT_RECORD_KIND
            or type(unit_record.get("schema_version")) is not int
            or unit_record.get("schema_version") != UNIT_RECORD_VERSION):
        raise ValueError(f"unit_record must be a {UNIT_RECORD_KIND} v{UNIT_RECORD_VERSION} "
                         "written by scripts/s2_run_units.py")
    if unit_record.get("scope") != scope:
        raise ValueError(f"unit_record scope {unit_record.get('scope')!r} differs from {scope!r}")
    if set(unit_record) != _UNIT_RECORD_KEYS:
        raise ValueError(f"unit_record must carry exactly {sorted(_UNIT_RECORD_KEYS)}; "
                         f"missing {sorted(_UNIT_RECORD_KEYS - set(unit_record))}, extra "
                         f"{sorted(set(unit_record) - _UNIT_RECORD_KEYS)}")
    if not _text(unit_record["job_id"]) or type(unit_record["steps"]) is not list or not all(
            type(step) is dict and _text(step.get("id")) for step in unit_record["steps"]):
        raise ValueError("unit_record requires a text job_id and a steps list of {id, ...}")
    exit_code = unit_record["compose_exit_code"]
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError("unit_record compose_exit_code must be an integer or null")
    for key in ("stop_run", "outcome"):
        if unit_record[key] is not None and not _text(unit_record[key]):
            raise ValueError(f"unit_record {key} must be text or null")


def _runner_interruption(unit_record):
    """The recorder's ``state["interruption"]`` re-derived from the unit record, or None.

    scripts/s2_run_units.py sets it to ``compose_up_timeout`` when the Compose call timed out
    (750, the ``compose_up`` step records ``timed_out``), to ``internal_error:<type>`` (537,
    overwriting) or ``interrupted:<type>`` (486, first wins) when an exception stopped the
    Compose call itself, before its step was recorded; ``stop_run`` keeps that reason. An
    interrupt after Compose returned (captures, teardown) is not an interruption (474-486).
    """
    steps = [step for step in unit_record["steps"] if step["id"] == "compose_up"]
    stop = unit_record["stop_run"]
    if len(steps) == 1:
        if unit_record["outcome"] == "internal_error" and _text(stop) and stop.startswith(
                "internal_error:"):
            return stop
        return "compose_up_timeout" if steps[0].get("timed_out") is True else None
    if not steps and _text(stop) and stop.startswith(("internal_error:", "interrupted:")):
        return stop
    return None


def _read(root, receipts, path, maximum, issues):
    """Hash-checked bytes of one receipted file, or None (absent or failed, with an issue)."""
    receipt = receipts.get(path)
    if receipt is None:
        return None
    if receipt["size_bytes"] > maximum:
        _issue(issues, "file_exceeds_bound", path,
               f"{receipt['size_bytes']} bytes exceed the {maximum}-byte read bound")
        return None
    try:
        return read_artifact(root, receipt)
    except (OSError, ValueError) as exc:
        _issue(issues, "receipt_mismatch", path,
               f"Receipted bytes could not be verified: {type(exc).__name__}: {exc}")
        return None


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _stream_values(table):
    """name -> list of (timestamp, value, valid) in written order (pure)."""
    streams = {}
    columns = table.select(["name", "timestamps_us", "values", "valid"]).to_pylist()
    for row in columns:
        streams.setdefault(row["name"], []).append(
            (row["timestamps_us"], row["values"], row["valid"]))
    return streams


def _full_max_deviation(samples):
    """(value, reason): full-stream maximum or None with the reason (never 0)."""
    if samples is None:
        return None, "stream_absent"
    if not samples:
        return None, "stream_empty"
    values = [value for _stamp, value, _valid in samples]
    if any(not _finite_number(value) or value < 0 for value in values):
        return None, "null_nonfinite_or_negative_sample"
    return max(values), None


def _positive_count(samples):
    if samples is None or not samples:
        return None
    values = [value for _stamp, value, _valid in samples]
    if any(not _finite_number(value) for value in values):
        return None
    return sum(1 for value in values if value > 0)


def _metric_facts(data, *, clipgt_id, rollout_uuid, run_uuid, path, issues):
    """Inspection plus value facts for one metrics.parquet; None facts when unusable."""
    inspection = inspect_metrics_parquet(data, clipgt_id=clipgt_id, rollout_uuid=rollout_uuid,
                                         run_uuid=run_uuid, path=path)
    facts = {"inspection": inspection, "usable": False, "required_metrics_verified": False,
             "required_defects": [], "invalid_samples": None, "full_max_deviation_m": None,
             "deviation_unknown_reason": "metrics_unusable", "corridor_departure": None,
             "black_frame_count": None, "metric_names": None, "run_name": None,
             "raw_metric_rows": inspection["rows"]}
    codes = {item["code"] for item in inspection["issues"]}
    if "metrics_unreadable" in codes:
        _issue(issues, "metrics_unreadable", path, "metrics.parquet is not readable Parquet")
        return facts
    types = inspection["column_types"] or {}
    if not inspection["schema_matches"] or any(
            types.get(name) != kind for name, kind in _PINNED_COLUMN_TYPES.items()):
        _issue(issues, "metrics_schema_differs", path,
               "Columns or Arrow types differ from the pinned writer (data.py:2027-2077); "
               "no value from this file is used")
        return facts
    if not all(inspection["identity_matches"].values()):
        _issue(issues, "metrics_identity_mismatch", path,
               "metrics identity columns differ from the folder and run_metadata binding: "
               f"{sorted(k for k, ok in inspection['identity_matches'].items() if not ok)}")
        return facts
    table = pq.read_table(pa.BufferReader(data), use_threads=False)
    streams = _stream_values(table)
    facts["usable"] = True
    facts["metric_names"] = sorted(str(name) for name in streams)
    run_names = inspection["identity"]["run_name"]
    facts["run_name"] = run_names[0] if len(run_names) == 1 and _text(run_names[0]) else None
    by_stream = {name: item["valid_false"] for name, item in inspection["streams"].items()
                 if item["valid_false"]}
    null_valid = {name: item["valid_null"] for name, item in inspection["streams"].items()
                  if item["valid_null"]}
    facts["invalid_samples"] = {
        "total": sum(by_stream.values()), "by_stream": dict(sorted(by_stream.items())),
        "required": {name: count for name, count in sorted(by_stream.items())
                     if name in RAW_AGGREGATIONS},
        "valid_null_by_stream": dict(sorted(null_valid.items())),
    }
    required = inspection["required_streams"]
    defects = [f"missing:{name}" for name in required["missing"]]
    defects += [f"aggregation:{name}" for name in required["aggregation_mismatch"]]
    for name in required["present"]:
        item = inspection["streams"][name]
        coverage = required["coverage_of_required_union"][name]
        for label, bad in (("valid_false", item["valid_false"]),
                           ("valid_null", item["valid_null"]),
                           ("values_null", item["values_null"]),
                           ("values_nonfinite", item["values_nonfinite"]),
                           ("timestamps_null", item["timestamps_null"]),
                           ("duplicate_timestamps", item["duplicate_timestamps"]),
                           ("not_strictly_increasing", not item["strictly_increasing"]),
                           ("partial_coverage", coverage["covered"] < coverage["union"])):
            if bad:
                defects.append(f"{label}:{name}")
    if facts["run_name"] is None:
        defects.append("run_name_not_unique")
    facts["required_defects"] = sorted(defects)
    facts["required_metrics_verified"] = not defects
    deviation, reason = _full_max_deviation(streams.get(DEVIATION_STREAM))
    facts["full_max_deviation_m"], facts["deviation_unknown_reason"] = deviation, reason
    corridor = _positive_count(streams.get(CORRIDOR_STREAM))
    facts["corridor_departure"] = None if corridor is None else corridor > 0
    facts["black_frame_count"] = _positive_count(streams.get(BLACK_STREAM))
    return facts


def _summary_valid(summary):
    """gate.s2_capture's own condition (s2_capture.py:758-759)."""
    return (type(summary) is dict and summary.get("schema_version") == 1
            and type(summary.get("rollouts")) is list)


def _summary_rows(summary):
    return summary["rollouts"] if _summary_valid(summary) else []


def _eval_config_facts(root, by_path, maximum, issues):
    """(fail_on_driver_termination or None, receipt): the horizon attestation input."""
    raw = _read(root, by_path, EVAL_CONFIG_PATH, maximum, issues)
    receipt = by_path.get(EVAL_CONFIG_PATH)
    if raw is None:
        if receipt is None:
            _issue(issues, "eval_config_absent", EVAL_CONFIG_PATH,
                   "No eval-config.yaml; a driver-ended session cannot be excluded, so "
                   "requested_steps/actual_steps are withheld", "warning")
        return None, receipt
    try:
        loaded = yaml.load(raw, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except _YAML_LOAD_ERRORS as exc:
        # YAML that parses can still fail to construct (``when: 2026-13-45`` raises
        # ValueError from datetime.date) or exhaust the recursion limit (deep nesting); each
        # is the same documented blocking issue, never an escaping exception.
        _issue(issues, "eval_config_unreadable", EVAL_CONFIG_PATH,
               f"eval-config.yaml could not be parsed or constructed as YAML: "
               f"{type(exc).__name__}")
        return None, receipt
    flag = loaded.get("fail_on_driver_termination") if type(loaded) is dict else None
    flag = flag if type(flag) is bool else None
    if flag is not True:
        _issue(issues, "driver_termination_not_fatal", EVAL_CONFIG_PATH,
               f"fail_on_driver_termination is {flag!r}, not true: a driver-ended session "
               "still completes, so requested_steps/actual_steps are withheld", "warning")
    return flag, receipt


def _ledger_cross_check(job, ledger_attempts, expected, expected_ref, interruption, issues):
    """Ledger attempts of the job must be exactly the re-derived terminal events."""
    terminal = [event for event in expected if event["type"] != "started"]
    observed = [(item["attempt_id"], item["state"], item["terminal_reason"],
                 item["upstream_identity"]) for item in ledger_attempts]
    derived = [(event["attempt_id"], _STATE_FOR_EVENT[event["type"]], event["reason"],
                event["upstream_identity"]) for event in terminal]
    if observed != derived:
        _issue(issues, "ledger_attempts_differ_from_evidence", f"job:{job['job_id']}",
               f"Ledger attempts {observed!r} differ from the attempts the receipted evidence "
               f"implies {derived!r} (runner interruption {interruption!r})")
        return False
    for item in ledger_attempts:
        ends = [event for event in item["events"]
                if type(event) is dict and event.get("type") != "started"]
        refs = ends[-1].get("evidence_refs") if ends else None
        if type(refs) is not list or expected_ref not in refs:
            _issue(issues, "ledger_event_evidence_unbound", f"attempt:{item['attempt_id']}",
                   "The terminal ledger event does not cite the re-derived attempt evidence")
            return False
    return True


def extract_unit(root, receipts, unit_record, ledger_report, *, scope,
                 max_parquet_bytes=MAX_PARQUET_BYTES, max_log_bytes=MAX_LOG_BYTES,
                 max_config_bytes=MAX_CONFIG_BYTES):
    """Return one unit's observations and adapter sidecar from receipted outputs (L1/L2).

    Structural misuse (scope, receipts, unit record kind/scope, ledger shape, a synthetic
    or diagnostic unit or job, a job absent from the ledger) raises ``ValueError``; data
    defects are issues.
    ``status``: ``blocked`` on any blocking issue (integrity, binding, schema), else
    ``unverified`` when the summary or every Compose log is absent, else ``passed``.
    ``passed`` is extraction completeness only, never admission.
    """
    _check_scope(scope)
    _check_receipts(receipts)
    _check_unit_record(unit_record, scope)
    _check_ledger(ledger_report)
    job_id = unit_record["job_id"]
    jobs = [job for job in ledger_report["jobs"] if job.get("job_id") == job_id]
    if len(jobs) != 1:
        raise ValueError(f"job {job_id!r} must occur exactly once in the ledger")
    job = jobs[0]
    attempt_by_id = {item["attempt_id"]: item for item in ledger_report["attempts"]
                     if type(item) is dict and item.get("job_id") == job_id}
    ledger_attempts = [attempt_by_id[value] for value in job["attempt_ids"]
                       if value in attempt_by_id]
    issues = []
    by_path = {item["path"]: item for item in receipts}
    receipts_sha = identity_hash(sorted(receipts, key=lambda item: item["path"]))
    hash_steps = [step for step in unit_record["steps"] if step["id"] == RUN_DIR_HASH_STEP]
    recorded_sha = hash_steps[0].get("files_sha256") if len(hash_steps) == 1 else None
    recorded_status = hash_steps[0].get("status") if len(hash_steps) == 1 else None
    manifest_bound = recorded_status == "passed" and recorded_sha == receipts_sha
    if not manifest_bound:
        _issue(issues, "run_dir_manifest_unbound", "unit_record.steps.run_dir_hash",
               f"Receipts hash {receipts_sha} differs from the unit record's single passed "
               f"run_dir_hash files_sha256 {recorded_sha!r} (status {recorded_status!r}, "
               f"{len(hash_steps)} steps); no receipted byte is bound to the pre-teardown "
               "manifest")
    if unit_record["scene_id"] != job["scene_id"]:
        _issue(issues, "unit_record_scene_differs", "unit_record.scene_id",
               "Unit record scene differs from the ledger job scene")

    raw = _read(root, by_path, RUN_METADATA_PATH, max_config_bytes, issues)
    binding = bind_run_uuid(raw) if raw is not None else None
    metadata_text = None
    if raw is not None:
        try:
            metadata_text = raw.decode("utf-8", "strict")
        except UnicodeError:
            metadata_text = None
    if binding is None:
        if RUN_METADATA_PATH not in by_path:
            _issue(issues, "run_metadata_absent", RUN_METADATA_PATH,
                   "run_uuid cannot be bound without run_metadata.yaml")
    elif binding["status"] != "passed":
        _issue(issues, "run_uuid_unbound", RUN_METADATA_PATH,
               "; ".join(item["message"] for item in binding["issues"]))
    run_uuid = binding["run_uuid"] if binding and binding["status"] == "passed" else None

    logs, log_receipts = [], []
    for path, source in LOG_PATHS:
        data = _read(root, by_path, path, max_log_bytes, issues)
        if data is not None:
            logs.append({"source": source, "text": data.decode("utf-8", "replace")})
            log_receipts.append(by_path[path])
    summary = None
    data = _read(root, by_path, SUMMARY_PATH, max_log_bytes, issues)
    if data is not None:
        try:
            summary = read_json_bytes(data)
        except ValueError:
            summary = {"unreadable": True}
            _issue(issues, "summary_unreadable", SUMMARY_PATH, "results-summary.json is not JSON")
        else:
            if not _summary_valid(summary):
                _issue(issues, "summary_invalid", SUMMARY_PATH,
                       "results-summary.json is not schema_version 1 with a rollouts list")
    driver_termination, eval_receipt = _eval_config_facts(root, by_path, max_config_bytes,
                                                          issues)
    horizon_attested = driver_termination is True

    # The runner's reconcile input, rebuilt from the same receipted bytes
    # (scripts/s2_run_units.py _Run._record_attempts).
    evidence = {"scope": unit_record["scope"], "scene_id": job["scene_id"],
                "requested_steps": job["requested_steps"], "logs": logs,
                "rollout_files": sorted(by_path), "run_metadata": metadata_text,
                "summary": summary, "compose_exit_code": unit_record["compose_exit_code"]}
    report = reconcile_unit_attempts(evidence)
    expected_ref = EVIDENCE_REF_PREFIX + identity_hash(report)
    reconcile_steps = [step for step in unit_record["steps"]
                       if type(step) is dict and step.get("id") == "reconcile"]
    recorded_ref = reconcile_steps[0].get("evidence_ref") if len(reconcile_steps) == 1 else None
    evidence_bound = recorded_ref == expected_ref
    if not evidence_bound:
        _issue(issues, "attempt_evidence_unbound", "unit_record.steps.reconcile",
               f"Unit record evidence_ref {recorded_ref!r} differs from the evidence re-derived "
               f"from receipted bytes {expected_ref!r}")
    interruption = _runner_interruption(unit_record)
    derived_events = unit_terminal_events(job_id, report, 0, expected_ref,
                                          log_count=len(logs), interruption=interruption)
    ledger_bound = _ledger_cross_check(job, ledger_attempts, derived_events, expected_ref,
                                       interruption, issues)

    evidence_attempts = {item["rollout_uuid"]: item for item in report["attempts"]}
    rows = _summary_rows(summary)
    attempts, observations, out_of_envelope, without_identity = [], [], [], []
    invalid_total = 0
    for attempt in ledger_attempts:
        identity = attempt["upstream_identity"]
        entry = {"attempt_id": attempt["attempt_id"], "ledger_state": attempt["state"],
                 "terminal_reason": attempt["terminal_reason"], "upstream_identity": identity,
                 "evidence_state": None, "effective_steps": None, "folder": None,
                 "summary_rows": [], "metrics": None, "full_max_deviation_m": None,
                 "deviation_unknown_reason": None, "out_of_envelope": None,
                 "observation_emitted": False}
        attempts.append(entry)
        if identity is None:
            without_identity.append(attempt["attempt_id"])
            continue
        path = f"attempt:{attempt['attempt_id']}"
        if identity["clipgt_id"] != job["scene_id"]:
            _issue(issues, "scene_clip_mismatch", path,
                   f"clipgt_id {identity['clipgt_id']!r} differs from scene_id "
                   f"{job['scene_id']!r}")
        if run_uuid is None or identity["run_uuid"] != run_uuid:
            _issue(issues, "identity_run_uuid_mismatch", path,
                   "Ledger run_uuid differs from run_metadata.yaml")
        found = evidence_attempts.get(identity["rollout_id"])
        if found is None:
            _issue(issues, "attempt_without_evidence", path,
                   "No receipted log, folder or summary names this rollout")
            continue
        entry["evidence_state"] = found["state"]
        entry["effective_steps"] = list(found["effective_steps"])
        entry["folder"] = found["folder"]
        entry["summary_rows"] = list(found["summary_rows"])
        folder = found["folder"] or {}
        facts = None
        metrics_receipt = None
        if folder.get("metrics"):
            metrics_path = (f"rollouts/{identity['clipgt_id']}/{identity['rollout_id']}/"
                            "metrics.parquet")
            metrics_receipt = by_path.get(metrics_path)
            data = _read(root, by_path, metrics_path, max_parquet_bytes, issues)
            if data is not None:
                facts = _metric_facts(data, clipgt_id=identity["clipgt_id"],
                                      rollout_uuid=identity["rollout_id"],
                                      run_uuid=run_uuid, path=metrics_path, issues=issues)
        if facts is not None:
            entry["metrics"] = {key: facts[key] for key in (
                "usable", "required_metrics_verified", "required_defects", "invalid_samples",
                "corridor_departure", "black_frame_count", "metric_names", "run_name",
                "raw_metric_rows")}
            entry["metrics"]["streams"] = facts["inspection"]["streams"]
            entry["metrics"]["required_streams"] = facts["inspection"]["required_streams"]
            entry["full_max_deviation_m"] = facts["full_max_deviation_m"]
            entry["deviation_unknown_reason"] = facts["deviation_unknown_reason"]
            if facts["invalid_samples"] is not None:
                invalid_total += facts["invalid_samples"]["total"]
        else:
            entry["deviation_unknown_reason"] = "no_metrics_file"
        deviation = entry["full_max_deviation_m"]
        if deviation is not None:
            entry["out_of_envelope"] = deviation >= ENVELOPE_M
            if entry["out_of_envelope"]:
                out_of_envelope.append(attempt["attempt_id"])
        completion = found["state"] == "complete" and attempt["state"] == "completed"
        provenance = identity_hash({
            "attempt_evidence": expected_ref, "metrics": metrics_receipt,
            "run_metadata": by_path.get(RUN_METADATA_PATH),
            "summary": by_path.get(SUMMARY_PATH), "logs": log_receipts,
            "eval_config": eval_receipt, "run_dir_files_sha256": receipts_sha})
        row = {"attempt_id": attempt["attempt_id"], "job_id": job_id,
               **{key: identity[key] for key in IDENTITY_FIELDS},
               "completion_validated": completion,
               "required_metrics_verified": bool(facts and facts["required_metrics_verified"]),
               "provenance_ref": PROVENANCE_PREFIX + provenance,
               "full_max_deviation_m": deviation}
        if horizon_attested and len(found["effective_steps"]) == 1:
            row["requested_steps"] = job["requested_steps"]
            row["actual_steps"] = found["effective_steps"][0]
        if facts is not None and facts["usable"]:
            row["metric_names"] = facts["metric_names"]
            row["raw_metric_rows"] = facts["raw_metric_rows"]
            if facts["run_name"] is not None:
                row["run_name"] = facts["run_name"]
            if facts["corridor_departure"] is not None:
                row["corridor_departure"] = facts["corridor_departure"]
            if facts["black_frame_count"] is not None:
                row["black_frame_count"] = facts["black_frame_count"]
        observations.append(row)
        entry["observation_emitted"] = True
        # Summary vs full-stream facts for this rollout's scored rows (never used as values).
        for summary_row in rows:
            if (type(summary_row) is not dict
                    or summary_row.get("rollout_id") != identity["rollout_id"]
                    or type(summary_row.get("metrics")) is not dict
                    or "error" in summary_row["metrics"]):
                continue
            metrics = summary_row["metrics"]
            reported = metrics.get(DEVIATION_STREAM)
            if (deviation is not None and _finite_number(reported) and reported < deviation):
                _issue(issues, "summary_deviation_below_full_stream", path,
                       f"Summary {DEVIATION_STREAM} {reported} < full-stream maximum "
                       f"{deviation} (upstream truncation); the full maximum is used",
                       "info")
            if (deviation is not None and _finite_number(reported) and reported > deviation):
                _issue(issues, "summary_deviation_above_full_stream", path,
                       f"Summary {DEVIATION_STREAM} {reported} > full-stream maximum "
                       f"{deviation}: upstream aggregates a subset of the same samples, so "
                       "parquet and summary disagree")
            if facts is not None and facts["usable"]:
                missing = facts["inspection"]["required_streams"]["missing"]
                for name in missing:
                    if name in metrics:
                        _issue(issues, "summary_metric_without_raw_stream",
                               f"{path}.{name}",
                               f"Summary reports {name}={metrics[name]!r} but the raw stream "
                               "is absent (upstream zero-fill, processing.py:752-760); it is "
                               "not used", "warning")
    if without_identity:
        _issue(issues, "attempts_without_upstream_identity", f"job:{job_id}",
               f"{len(without_identity)} attempts have no upstream identity (pre-UUID "
               "failure); counted, no observation row", "info")
    if invalid_total:
        _issue(issues, "invalid_samples_preserved", f"job:{job_id}",
               f"{invalid_total} metric samples carry valid=False; kept and counted", "info")
    if out_of_envelope:
        _issue(issues, "out_of_envelope_attempts", f"job:{job_id}",
               f"{len(out_of_envelope)} attempts reach >= {ENVELOPE_M} m over the full stream; "
               "reported separately and kept", "info")
    null_rows = [row.get("rollout_id") for row in rows
                 if type(row) is dict and row.get("run_uuid") is None]
    if summary is None and SUMMARY_PATH not in by_path:
        _issue(issues, "summary_absent", SUMMARY_PATH, "No results-summary.json", "warning")
    if not logs:
        _issue(issues, "logs_absent", "physloop-capture",
               "No Compose log is receipted and readable", "warning")
    folders = [item["folder"] for item in report["attempts"] if item["folder"] is not None]
    identity_map = [
        {**attempt["upstream_identity"],
         **{key: job[key] for key in ("scene_id", "road_id", "stratum", "replicate_id",
                                      "policy_id")}}
        for attempt in ledger_attempts if attempt["upstream_identity"] is not None]
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    if _blocking(issues):
        status = "blocked"
    elif summary is None or not logs:
        status = "unverified"
    else:
        status = "passed"
    return {
        "artifact_kind": UNIT_ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "gate.s2_actual_extract", "function": "extract_unit",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {"receipts": identity_hash(sorted(receipts,
                                                          key=lambda item: item["path"])),
                         "unit_record": identity_hash(unit_record),
                         "ledger_report": identity_hash(ledger_report)},
        "units": {"deviation": "m", "timestamps": "microseconds", "size_bytes": "bytes",
                  "steps": "policy steps"},
        "bounds": {"max_parquet_bytes": max_parquet_bytes, "max_log_bytes": max_log_bytes,
                   "max_config_bytes": max_config_bytes},
        "status": status, "issues": issues,
        "admission": "not_applicable", "score_admitted": False, "runtime_qualified": False,
        "job": {key: job[key] for key in ("job_id", "phase", "arm", "policy_id", "scene_id",
                                          "road_id", "stratum", "replicate_id",
                                          "protocol_hash", "requested_steps", "status")},
        "run_uuid_binding": binding,
        "run_dir_manifest": {"receipts_files_sha256": receipts_sha,
                             "unit_record_files_sha256": recorded_sha,
                             "unit_record_status": recorded_status, "bound": manifest_bound},
        "horizon": {"eval_config": eval_receipt,
                    "fail_on_driver_termination": driver_termination,
                    "steps_reported": horizon_attested,
                    "actual_steps_source": "single Session STARTING steps= value (planned "
                                           "clipped horizon, event_loop.py:622-627)"},
        "attempt_evidence": {"reference": expected_ref, "unit_record_reference": recorded_ref,
                             "bound_to_unit_record": evidence_bound,
                             "bound_to_ledger": ledger_bound,
                             "runner_interruption": interruption,
                             "unit_outcome": report["unit_outcome"],
                             "status": report["status"],
                             "codes": sorted({item["code"] for item in report["issues"]}),
                             "internal_retries": len(report["internal_retries"]),
                             "pre_uuid_failures": len(report["pre_uuid_failures"]),
                             "compose_exit_code_used_for_success": False},
        "attempts": attempts,
        "observations": observations,
        "counts": {"ledger_attempts": len(ledger_attempts),
                   "observations": len(observations),
                   "attempts_without_upstream_identity": len(without_identity),
                   "invalid_samples": invalid_total,
                   "summary_rows": len(rows),
                   "summary_rows_null_run_uuid": len(null_rows),
                   "folders_without_complete": sum(1 for f in folders
                                                   if not f["complete_marker"])},
        "summary_rows_null_run_uuid": sorted(str(value) for value in null_rows),
        "out_of_envelope": {"threshold_m": ENVELOPE_M, "attempt_ids": sorted(out_of_envelope),
                            "comparison": "full_max_deviation_m >= threshold_m"},
        "normalize_inputs": {"summary": summary, "identity_map": identity_map,
                             "observations": observations, "arm": job["arm"],
                             "policy_id": job["policy_id"]},
    }


def _normalization_view(normalization):
    """JSON-native copy: adapter issue identities are tuples (gate.alpasim_adapter); lists
    here, following gate.s2_diagnostics._issue_view. Other values are kept unchanged, so a
    non-JSON value still fails closed in canonical_json_bytes."""
    view = dict(normalization)
    if type(view.get("issues")) is list:
        view["issues"] = [
            {**item, "identity": list(item["identity"])}
            if type(item) is dict and type(item.get("identity")) is tuple else item
            for item in view["issues"]]
    return view


def _hash_or_none(value):
    if value is None:
        return None
    try:
        return identity_hash(value)
    except (ValueError, TypeError):
        return None


def _conventions_problem(conventions):
    """(code, message) when conventions are unusable, else None."""
    if conventions is None:
        return ("owner_conventions_unregistered",
                "Huy's terminal and validity conventions are not supplied")
    if (type(conventions) is not dict or set(conventions) != CONVENTION_FIELDS
            or type(conventions["schema_version"]) is not int
            or conventions["schema_version"] != 1
            or not _text(conventions["conventions_id"])
            or not _text(conventions["source_ref"])
            or type(conventions["validity_definition"]) is not dict):
        return ("invalid_owner_conventions",
                f"Conventions require exactly {sorted(CONVENTION_FIELDS)} with schema_version 1")
    terminal = conventions["terminal_semantics"]
    if terminal in UNSUPPORTED_TERMINAL_SEMANTICS:
        return ("terminal_semantics_unsupported",
                (f"{terminal!r} cannot be honoured: the adapter requires actual_steps == "
                 "requested_steps (horizon_unverified); an adapter change is a shared change"))
    if terminal not in TERMINAL_SEMANTICS:
        return ("invalid_owner_conventions",
                f"terminal_semantics must be one of {TERMINAL_SEMANTICS}")
    return None


def assemble_batch(unit_extractions, ledger_report, *, scope, observed_evaluator_sha,
                   conventions):
    """Normalize every unit, describe validity and emit gate records only when admitted (L1).

    ``unit_extractions`` are ``extract_unit`` outputs over the same ``ledger_report``;
    ``observed_evaluator_sha`` is the AlpaSim checkout SHA the units ran (required; a
    caller argument, not bound to run evidence here, so labelled
    ``caller_supplied_not_authenticated``);
    ``conventions`` is Huy's object or ``None`` (required keyword, no default).
    """
    _check_scope(scope)
    _check_ledger(ledger_report)
    if type(unit_extractions) is not list or not all(
            type(item) is dict and item.get("artifact_kind") == UNIT_ARTIFACT_KIND
            and item.get("schema_version") == SCHEMA_VERSION for item in unit_extractions):
        raise ValueError(f"unit_extractions must be {UNIT_ARTIFACT_KIND} v{SCHEMA_VERSION} "
                         "artifacts")
    if not _text(observed_evaluator_sha):
        raise ValueError("observed_evaluator_sha is required (no default)")
    issues, reasons = [], []
    ledger_hash = identity_hash(ledger_report)
    by_job = {}
    for item in unit_extractions:
        job_id = item["job"]["job_id"]
        if item["scope"] != scope:
            _issue(issues, "unit_scope_differs", f"job:{job_id}", "Unit scope differs")
        if item["input_hashes"]["ledger_report"] != ledger_hash:
            _issue(issues, "unit_ledger_differs", f"job:{job_id}",
                   "Unit was extracted against a different ledger report")
        if job_id in by_job:
            _issue(issues, "duplicate_unit_extraction", f"job:{job_id}",
                   "Several extractions name one job")
            continue
        by_job[job_id] = item
    job_order = [job["job_id"] for job in ledger_report["jobs"]]
    for job_id in sorted(set(by_job) - set(job_order)):
        _issue(issues, "unit_extraction_unscheduled", f"job:{job_id}",
               "Extraction names no scheduled job")
    for job_id in job_order:
        if job_id not in by_job:
            _issue(issues, "unit_extraction_missing", f"job:{job_id}",
                   "Scheduled job has no unit extraction; it is not admitted as absent")
    ordered = [by_job[job_id] for job_id in job_order if job_id in by_job]

    observations, normalizations = [], []
    for item in ordered:
        job = item["job"]
        observations.extend(item["observations"])
        inputs = item["normalize_inputs"]
        if item["status"] != "passed":
            reasons.append(f"unit_extraction_{item['status']}:{job['job_id']}")
        if inputs["summary"] is None:
            reasons.append(f"summary_absent:{job['job_id']}")
            continue
        normalization = normalize_summary(
            inputs["summary"], inputs["identity_map"], inputs["observations"],
            arm=inputs["arm"], policy_id=inputs["policy_id"],
            expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
            observed_evaluator_sha=observed_evaluator_sha)
        if not normalization["admitted"]:
            reasons.append(f"normalization_not_admitted:{job['job_id']}")
        normalizations.append({"job_id": job["job_id"], "arm": job["arm"],
                               "protocol_hash": job["protocol_hash"],
                               "policy_id": job["policy_id"],
                               "normalization": normalization})

    problem = _conventions_problem(conventions)
    definition = None if problem is not None else conventions["validity_definition"]
    entries = [{key: item[key] for key in ("arm", "protocol_hash", "policy_id",
                                           "normalization")} for item in normalizations]
    validity = describe_validity(ledger_report, observations, entries or None,
                                 validity_definition=definition, scope=scope)
    if definition is not None and any(
            item["code"] == "invalid_validity_definition" for item in validity["issues"]):
        problem = ("invalid_owner_conventions",
                   "validity_definition is rejected by describe_validity")
        definition = None
    if not ledger_report["admitted"]:
        reasons.append("ledger_not_admitted")
    if validity["status"] != "passed":
        reasons.append("validity_report_blocked")
    if definition is not None:
        for unit in validity["units"]:
            if unit["validity"] != "valid":
                reasons.append(f"unit_validity_{unit['validity']}:{unit['job_id']}")
    if _blocking(issues):
        reasons.append("batch_inputs_inconsistent")
    metrics = [item["normalization"]["metric"] for item in normalizations]
    adapter_metric = metrics[0] if metrics and all(m == metrics[0] for m in metrics) else None
    if metrics and adapter_metric is None:
        reasons.append("adapter_metric_differs")
    if not normalizations:
        reasons.append("no_normalization")
    reasons = sorted(set(reasons))
    if problem is not None:
        reasons = [problem[0], *reasons]
    admitted = not reasons
    records = []
    if admitted:
        for item in normalizations:
            records.extend(item["normalization"]["records"])
    oe_attempts = sorted(row["attempt_id"] for row in observations
                         if _finite_number(row.get("full_max_deviation_m"))
                         and row["full_max_deviation_m"] >= ENVELOPE_M)
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    return {
        "artifact_kind": BATCH_ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "gate.s2_actual_extract", "function": "assemble_batch",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {"ledger_report": ledger_hash,
                         "unit_extractions": sorted(identity_hash(item)
                                                    for item in unit_extractions),
                         "conventions": _hash_or_none(conventions)},
        "units": {"deviation": "m", "records": "adapter records (one per admitted rollout)"},
        "status": "passed" if admitted else "blocked",
        "status_meaning": ("passed: every extraction, normalization and owner convention "
                           "admitted the batch; never a runtime qualification"),
        "admission": {"status": "admitted" if admitted else "blocked",
                      "code": None if admitted else reasons[0], "reasons": reasons},
        "conventions": None if conventions is None else {
            "conventions_id": conventions.get("conventions_id")
            if type(conventions) is dict else None,
            "sha256": _hash_or_none(conventions), "status": "supplied_not_authenticated",
            "problem": None if problem is None else {"code": problem[0],
                                                     "message": problem[1]}},
        "observed_evaluator_sha": {"value": observed_evaluator_sha,
                                   "status": "caller_supplied_not_authenticated"},
        "observations": observations,
        "normalizations": [{**item, "normalization": _normalization_view(item["normalization"])}
                           for item in normalizations],
        "validity_report": validity,
        "out_of_envelope": {"threshold_m": ENVELOPE_M, "attempt_ids": oe_attempts,
                            "records": sum(1 for row in records
                                           if row["deviation_m"] >= ENVELOPE_M),
                            "handling": "kept in observations and records; reported "
                                        "separately"},
        "gate_inputs": {"records": records, "adapter_metric": adapter_metric if admitted
                        else None},
        "score_admitted": admitted, "runtime_qualified": False,
        "issues": issues,
    }
