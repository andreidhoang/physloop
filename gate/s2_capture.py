"""Inspect pinned ASL receipts and per-unit attempt evidence; never admit scores.

Tickets: S2 CPU handoff E2; build-spec A4 C1/A1 (explicit operator bound with
streaming reads), C2 (attempt evidence from Compose log lines plus a rollout
folder scan), C3 (exit code is never success evidence), C4 (ASL carries
scene/session, not run_uuid) and P2 (run_uuid bound from run_metadata.yaml).
Serves S2.1-S2.7 [C] (G1, G2, G3-R1, G7).

The upstream evaluator binds metadata.scene_id to clipgt_id and metadata.session_uuid
to rollout_id. run_uuid comes from external run metadata, not the ASL session UUID.
Receipt checks establish byte integrity, not producer authenticity or completion.

Verification level per function (build-spec section 1):

- ``inspect_capture``: L1 receipt/attempt binding plus L2 ASL parsing.
- ``parse_runtime_log``: L2. Message formats are pinned AlpaSim e121e591 source
  lines (cited at each pattern); Docker Compose line prefixes and service names
  are ``qualify_on_host``.
- ``bind_run_uuid``: L2 against the pinned PyYAML writer of run_metadata.yaml.
- ``scan_rollout_files``: L2 against pinned rollout path construction.
- ``reconcile_unit_attempts``: L1 over parsed evidence; L3 for real captures.

Size bound: ``MAX_ASL_BYTES`` equals ``gate.s2_asl.MAX_FILE_BYTES`` (2 GiB), an
agent technical default (D-018), provisional until the first measured ASL. ASL
files are streamed through a no-follow, hash-while-parse reader; at most one
frame payload (64 MiB) plus the decoded report is held in RAM. A hash or size
mismatch discards the inspection and blocks the capture.

This module never: launches, stops or inspects containers; reads the host
filesystem except receipted files below ``root`` (and its own source, hashed
once at import for producer identity); parses YAML (no YAML
dependency is locked, so run metadata, generated configs and summaries arrive
already parsed except for the strict run_uuid line extractor); infers success
from an exit code; selects among retried attempts; admits scores.
"""

import hashlib
import os
import re
import stat
from pathlib import Path

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA
from gate.s2_asl import MAX_FILE_BYTES, inspect_asl_stream
from gate.s2_identity import (
    _directory_fd,
    canonical_json_bytes,
    identity_hash,
    read_artifact,
    read_json_bytes,
)
from gate.s2_ledger import reconcile_attempts
from gate.s2_schedule import build_schedule

REQUEST_FIELDS = {
    "schema_version", "evidence_kind", "capture_origin", "evaluator_sha",
    "schedule", "schedule_request", "events", "artifacts",
}
# Agent technical default (D-018), provisional until the first measured ASL.
MAX_ASL_BYTES = MAX_FILE_BYTES
MAX_EVENTS_BYTES = 16 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024
# Shared contract rule 2.1 scopes for emitted artifacts.
SCOPES = ("synthetic", "public_source_fixture", "actual")


def _read_source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


# Hashed once at import; reconciliation itself performs no file I/O.
_SOURCE_SHA256 = _read_source_sha256()


def _issue(report, code, path, message, *, severity="blocking", evidence_refs=()):
    report["issues"].append({
        "code": code, "path": path, "severity": severity,
        "evidence_refs": list(evidence_refs), "message": message,
    })


def _blocking(issues):
    return any(item.get("severity", "blocking") == "blocking" for item in issues)


def _bounded_read(root, receipt, maximum):
    if (
        type(receipt) is not dict or type(receipt.get("size_bytes")) is not int
        or not 0 <= receipt["size_bytes"] <= maximum
    ):
        raise ValueError(f"Artifact exceeds or lacks the declared byte limit ({maximum})")
    return read_artifact(root, receipt)


class _VerifyingReader:
    """Count and hash every byte the inspector reads; never read past size+1."""

    def __init__(self, handle, expected_size):
        self._handle = handle
        self._remaining = expected_size + 1
        self._hasher = hashlib.sha256()
        self.size = 0

    def read(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("Bounded reads require a nonnegative byte count")
        count = min(count, _STREAM_CHUNK_BYTES, self._remaining)
        data = self._handle.read(count) if count else b""
        self._remaining -= len(data)
        self.size += len(data)
        self._hasher.update(data)
        return data

    def drain(self):
        while self.read(_STREAM_CHUNK_BYTES):
            pass

    def hexdigest(self):
        return self._hasher.hexdigest()


def _open_receipted(root, receipt):
    """Open one receipted regular file below root without following any symlink.

    Mirrors ``gate.s2_identity.read_artifact`` validation and descriptor-relative
    no-follow opens, but returns the open descriptor for streaming verification.
    """
    if type(receipt) is not dict or set(receipt) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Artifact receipt requires exactly path, sha256 and size_bytes")
    name, expected_hash, expected_size = (
        receipt["path"], receipt["sha256"], receipt["size_bytes"])
    if type(name) is not str or not name or "\x00" in name:
        raise ValueError("Artifact path must be a nonempty relative POSIX path")
    components = name.split("/")
    if any(part in ("", ".", "..") for part in components):
        raise ValueError("Artifact path must be relative without traversal or empty components")
    if type(expected_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("Artifact sha256 must be 64 lowercase hexadecimal characters")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("Artifact size_bytes must be a nonnegative integer")
    directory = _directory_fd(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        for component in components[:-1]:
            following = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = following
        descriptor = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                             | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Artifact must be a regular file")
        if metadata.st_size != expected_size:
            raise ValueError("Artifact size does not match receipt")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _inspect_receipted_asl(root, receipt, maximum, **arguments):
    """Stream one ASL through hash verification; raise unless bytes match receipt."""
    if (
        type(receipt) is not dict or type(receipt.get("size_bytes")) is not int
        or not 0 <= receipt["size_bytes"] <= maximum
    ):
        raise ValueError(f"Artifact exceeds or lacks the declared byte limit ({maximum})")
    descriptor = _open_receipted(root, receipt)
    try:
        handle = os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        reader = _VerifyingReader(handle, receipt["size_bytes"])
        inspection = inspect_asl_stream(reader, max_file_bytes=maximum, **arguments)
        reader.drain()
    if reader.size != receipt["size_bytes"]:
        raise ValueError("Artifact size changed while reading")
    if reader.hexdigest() != receipt["sha256"]:
        raise ValueError("Artifact checksum does not match receipt")
    return inspection


def inspect_capture(root, request, *, max_asl_bytes=MAX_ASL_BYTES):
    """Return every ledger attempt and available ASL inspection, including failures.

    ``capture_inspection_passed`` requires both ledger admission and successful
    parsing/binding of all expected logs, with no blocking issue. Warnings (for
    example effective steps below requested) are retained and do not block. It
    is strictly a structural check: completion, render/map evidence and actual
    scorer execution remain unqualified even when this field is true.

    ``max_asl_bytes`` is an operator-supplied bound, never a limit authorized by
    the artifact's own receipt. Receipts above it are rejected before opening.
    """
    report = {
        "schema_version": 1,
        "evidence_kind": "alpasim_asl_inspection_v1",
        "capture_origin": None,
        "capture_inspection_passed": False,
        "ledger": None,
        "captures": [],
        "issues": [],
        "request": None,
        "request_sha256": None,
        "score_admitted": False,
        "runtime_qualified": False,
        "cloud_launch_authorized": False,
        "provenance_authenticated": False,
        "max_asl_bytes": max_asl_bytes if type(max_asl_bytes) is int else None,
        "max_asl_bytes_basis": "operator_bound_agent_default_D-018_provisional",
        "asl_read_mode": "streamed_hash_verified_no_follow",
        "max_events_bytes": MAX_EVENTS_BYTES,
        "remaining": [
            "actual completion and requested/effective horizon reconciliation",
            "complete renderer, black-frame and map evidence",
            "raw metrics and actual scorer-path qualification",
        ],
    }

    try:
        if type(max_asl_bytes) is not int or max_asl_bytes <= 0:
            raise ValueError("max_asl_bytes must be a positive integer")
        request = read_json_bytes(canonical_json_bytes(request))
        report["request"] = request
        report["request_sha256"] = identity_hash(request)
        if type(request) is not dict or set(request) != REQUEST_FIELDS:
            raise ValueError("Expected the exact ASL inspection request fields")
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Unsupported inspection schema version")
        if request["evidence_kind"] != "alpasim_asl_inspection_v1":
            raise ValueError("ASL inspection is distinct from synthetic score ingestion")
        if request["capture_origin"] not in ("synthetic_fixture", "runtime_claim_unverified"):
            raise ValueError("Origin must be synthetic_fixture or runtime_claim_unverified")
        report["capture_origin"] = request["capture_origin"]
        if request["evaluator_sha"] != SUPPORTED_EVALUATOR_SHA:
            raise ValueError("Unsupported evaluator/source pin")
        schedule = request["schedule"]
        if request["schedule_request"] is not None:
            if type(request["schedule_request"]) is not dict:
                raise ValueError("schedule_request must be an object")
            if build_schedule(**request["schedule_request"]) != schedule:
                raise ValueError("Schedule differs from its complete source request")
        elif not (
            type(schedule) is list and schedule
            and all(type(job) is dict and job.get("phase") == "synthetic" for job in schedule)
        ):
            raise ValueError("Pilot/full schedules require their source request")
        if type(request["artifacts"]) is not list:
            raise ValueError("artifacts must be a list")
        events = read_json_bytes(_bounded_read(root, request["events"], MAX_EVENTS_BYTES))
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_capture_request", "request", str(exc))
        return report

    ledger = report["ledger"] = reconcile_attempts(schedule, events)
    # Do not short-circuit on ledger failure: partial and retried captures still
    # provide useful diagnostics, and all original receipts remain in the ledger.
    jobs = {job["job_id"]: job for job in ledger["jobs"]}
    attempts = {attempt["attempt_id"] for attempt in ledger["attempts"]}
    artifacts, paths = {}, set()
    for index, artifact in enumerate(request["artifacts"]):
        path = f"artifacts[{index}]"
        if (
            type(artifact) is not dict or set(artifact) != {"attempt_id", "asl"}
            or type(artifact["attempt_id"]) is not str or not artifact["attempt_id"].strip()
        ):
            _issue(report, "invalid_attempt_artifact", path, "Expected attempt_id and ASL receipt")
            continue
        attempt_id = artifact["attempt_id"]
        if attempt_id not in attempts:
            _issue(report, "unknown_attempt_artifact", path, "Artifact has no recorded attempt")
            continue
        if attempt_id in artifacts:
            _issue(report, "duplicate_attempt_artifact", path,
                   "Only one ASL per attempt is supported")
            continue
        artifacts[attempt_id] = artifact["asl"]
        receipt = artifact["asl"]
        filename = receipt.get("path") if type(receipt) is dict else None
        if type(filename) is str:
            if filename in paths:
                _issue(report, "reused_asl_path", path,
                       "One ASL path cannot represent two attempts")
            paths.add(filename)

    asl_identity_owners = {}
    for attempt in ledger["attempts"]:
        attempt_id = attempt["attempt_id"]
        path = f"attempt:{attempt_id}"
        capture = {
            "attempt_id": attempt_id, "job_id": attempt["job_id"],
            "state": attempt["state"], "status": "unverified", "inspection": None,
        }
        report["captures"].append(capture)
        identity = attempt["upstream_identity"]
        if identity is None:
            capture["status"] = "no_upstream_identity"
            if attempt_id in artifacts:
                _issue(report, "unbound_asl", path,
                       "ASL cannot bind to an attempt without upstream IDs")
            continue
        # ASL carries scene/session, not the evaluator's external run_uuid.
        # Distinct run IDs therefore cannot disambiguate a reused ASL binding.
        asl_identity = (identity["clipgt_id"], identity["rollout_id"])
        owner = asl_identity_owners.setdefault(asl_identity, attempt_id)
        if owner != attempt_id:
            _issue(report, "reused_asl_identity", path,
                   f"ASL scene/session identity is already bound to attempt {owner!r}")
        # Keep inspecting both artifacts even when their binding is ambiguous.
        job = jobs.get(attempt["job_id"])
        if job is None:
            _issue(report, "unknown_capture_job", path, "Attempt is absent from the schedule")
            continue
        if job["scene_id"] != identity["clipgt_id"]:
            _issue(report, "scene_identity_mismatch", path,
                   "Schedule scene differs from clipgt_id")
            continue
        if attempt_id not in artifacts:
            capture["status"] = "missing_asl"
            _issue(report, "missing_asl", path,
                   "An attempt with upstream IDs requires its ASL receipt")
            continue
        try:
            inspection = _inspect_receipted_asl(
                root, artifacts[attempt_id], max_asl_bytes,
                expected_session_uuid=identity["rollout_id"],
                expected_scene_id=identity["clipgt_id"], requested_steps=job["requested_steps"],
            )
            capture["inspection"] = inspection
            capture["status"] = "verified" if inspection["inspection_valid"] else "invalid_asl"
            if not inspection["inspection_valid"]:
                _issue(report, "asl_inspection_failed", path, "See retained parser diagnostics")
            for item in inspection.get("issues", []):
                if item.get("severity") == "warning":
                    _issue(report, item["code"], path, item["message"], severity="warning",
                           evidence_refs=[f"{path}.inspection"])
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            capture["inspection"] = None
            capture["status"] = "artifact_unverified"
            _issue(report, "artifact_unverified", path, str(exc))
    report["capture_inspection_passed"] = ledger["admitted"] and not _blocking(report["issues"])
    return report


# --- C2/C3/P2: per-unit attempt evidence ------------------------------------------------

# Pinned formatters (AlpaSim e121e591):
#   worker:    "%(asctime)s.%(msecs)03d [W{id}] %(levelname)s:\t%(message)s", datefmt %H:%M:%S
#              (src/runtime/alpasim_runtime/worker/main.py:341-344)
#   scheduler: "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
#              (src/runtime/alpasim_runtime/simulate/__main__.py:296-300)
# Docker Compose adds "<service> | " and, for `logs --timestamps`, an RFC 3339
# timestamp; both prefixes are qualify_on_host (not AlpaSim source).
_LOG_PREFIX = re.compile(
    r"^(?:(?P<service>[A-Za-z0-9][A-Za-z0-9._-]*)\s+\|\s?)?"
    r"(?:(?P<docker_ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))\s)?"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) (?:\[W(?P<worker>\d+)\] )?"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL):\t(?P<message>.*)$"
)
_RUNTIME_SERVICE = re.compile(r"runtime-0(?:-\d+)?")
# Any "<token> |" Compose prefix, used only to count runtime-service lines (including
# traceback continuations, which carry no time/level) and to name an unknown service.
_SERVICE_PREFIX = re.compile(r"^(?P<service>[A-Za-z0-9][A-Za-z0-9._-]*)\s+\|")
_ROLLOUT_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_SCENE = r"[A-Za-z0-9._-]+"
_MESSAGES = (
    # event_loop.py:623-628 logger.info("Session STARTING: uuid=%s scene=%s steps=%d", ...)
    ("session_started", re.compile(
        rf"^Session STARTING: uuid=(?P<uuid>{_ROLLOUT_UUID}) scene=(?P<scene>{_SCENE}) "
        r"steps=(?P<steps>\d+)$")),
    # event_loop.py:688-698 "Session COMPLETED: uuid=%s scene=%s simulated %.2f sim seconds ..."
    ("session_completed", re.compile(
        rf"^Session COMPLETED: uuid=(?P<uuid>{_ROLLOUT_UUID}) scene=(?P<scene>{_SCENE}) "
        r"simulated ")),
    # worker/main.py:167-174 "Rollout FAILED: job=%s scene=%s uuid=%s retryable=%s error=%s\n%s"
    # scene/uuid are "N/A" when no rollout was allocated (worker/main.py:170-171).
    ("rollout_failed", re.compile(
        rf"^Rollout FAILED: job=(?P<job>[0-9a-f]{{32}}) scene=(?P<scene>{_SCENE}|N/A) "
        rf"uuid=(?P<uuid>{_ROLLOUT_UUID}|N/A) retryable=(?P<retryable>True|False) "
        r"error=(?P<error>.*)$")),
    # daemon/scheduler.py:203-212 "Retrying rollout after retryable infrastructure failure: "
    # "scene=%s failed_job=%s retry_job=%s attempt=%d/%d error=%s"; job IDs are uuid4().hex.
    ("retry_scheduled", re.compile(
        rf"^Retrying rollout after retryable infrastructure failure: scene=(?P<scene>{_SCENE}) "
        r"failed_job=(?P<failed_job>[0-9a-f]{32}) retry_job=(?P<retry_job>[0-9a-f]{32}) "
        r"attempt=(?P<attempt>\d+)/(?P<max_attempts>\d+) error=(?P<error>.*)$")),
)
_LOG_SOURCES = ("compose_up_stdout", "compose_logs", "runtime_worker_log")
_MARKERS = ("Session STARTING:", "Session COMPLETED:", "Rollout FAILED:", "Retrying rollout")


def parse_runtime_log(text, *, source):
    """Extract pinned attempt events from one captured runtime log text.

    ``source`` is ``compose_up_stdout`` or ``compose_logs`` (lines must carry a
    Compose service prefix; only the runtime service counts) or
    ``runtime_worker_log`` (``<RUN_DIR>/txt-logs/runtime_worker_<id>.log``, no
    prefix). Events keep 1-based line numbers. A line that names a marker but
    does not match its pinned format is a blocking ``unparsed_marker_line``; for
    Compose sources this includes any line containing a marker substring whose
    prefix is unrecognised (ANSI colour, a project-prefixed container name, no
    prefix) or names another service, so an unqualified host prefix fails
    closed instead of hiding attempts (review A4-F1). ``runtime_service_lines``
    counts lines attributable to the runtime (Compose: runtime service prefix;
    worker log: pinned formatter prefix); ``reconcile_unit_attempts`` blocks a
    capture where it is zero.
    Only the first line of a multi-line ``Rollout FAILED`` record is parsed; its
    traceback continuation lines carry no prefix and are ignored. Format checks
    cannot authenticate a line: an exception message that embeds a well-formed
    log line is indistinguishable by format alone, so folders, markers and
    summary rows corroborate log events in ``reconcile_unit_attempts``.
    """
    report = {"source": source, "events": [], "issues": [], "lines": 0,
              "ignored_lines": 0, "other_service_lines": 0, "runtime_service_lines": 0}
    if source not in _LOG_SOURCES:
        _issue(report, "invalid_log_source", "source", f"Supported sources: {_LOG_SOURCES}")
        return report
    if type(text) is not str:
        _issue(report, "invalid_log_text", "text", "Runtime log must be decoded text")
        return report
    lines = text.split("\n")
    report["lines"] = len(lines)
    compose = source != "runtime_worker_log"
    for number, raw in enumerate(lines, 1):
        line = raw.removesuffix("\r")
        has_marker = any(marker in line for marker in _MARKERS)
        if compose:
            service_prefix = _SERVICE_PREFIX.match(line)
            runtime = (service_prefix is not None
                       and _RUNTIME_SERVICE.fullmatch(service_prefix["service"]) is not None)
            if runtime:
                report["runtime_service_lines"] += 1
        prefix = _LOG_PREFIX.match(line)
        if prefix is None:
            report["ignored_lines"] += 1
            if (compose and has_marker) or (
                    not compose and line.lstrip().startswith(_MARKERS)):
                _issue(report, "unparsed_marker_line", f"line:{number}",
                       "Line names an attempt marker outside the pinned log prefix "
                       f"(observed start {line[:40]!r}; Compose prefix is qualify_on_host)")
            continue
        service = prefix["service"]
        if not compose:
            if service is not None:
                report["ignored_lines"] += 1
                continue
            report["runtime_service_lines"] += 1
        elif service is None or _RUNTIME_SERVICE.fullmatch(service) is None:
            report["other_service_lines" if service else "ignored_lines"] += 1
            if has_marker:
                _issue(report, "unparsed_marker_line", f"line:{number}",
                       f"Attempt marker under service {service!r}, not the runtime service "
                       "(Compose service naming is qualify_on_host)")
            continue
        message = prefix["message"]
        for kind, pattern in _MESSAGES:
            match = pattern.match(message)
            if match:
                event = {"kind": kind, "line": number, "level": prefix["level"],
                         "worker": None if prefix["worker"] is None else int(prefix["worker"]),
                         **{key: value for key, value in match.groupdict().items()
                            if key != "error"}}
                if "steps" in event:
                    event["steps"] = int(event["steps"])
                if kind == "rollout_failed":
                    event["retryable"] = event["retryable"] == "True"
                    event["error_first_line"] = match["error"]
                if kind == "retry_scheduled":
                    event["attempt"] = int(event["attempt"])
                    event["max_attempts"] = int(event["max_attempts"])
                report["events"].append(event)
                break
        else:
            if any(message.startswith(marker) for marker in _MARKERS):
                _issue(report, "unparsed_marker_line", f"line:{number}",
                       "Attempt marker does not match its pinned message format")
    return report


# configuration.py:274-305 writes run_metadata.yaml via yaml.dump (utils.py:31-43,
# default sort_keys) with "run_uuid": str(uuid.uuid4()); PyYAML emits it plain.
_RUN_UUID_LINE = re.compile(
    r"^run_uuid: ['\"]?(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"['\"]?$")


def bind_run_uuid(run_metadata_bytes):
    """Strictly extract the top-level run_uuid from run_metadata.yaml bytes.

    Exactly one unindented ``run_uuid:`` line is accepted, with a canonical
    lowercase 8-4-4-4-12 value (optionally single/double quoted). Anything else
    returns ``status: blocked``. This is not a YAML parser.
    """
    result = {"status": "blocked", "run_uuid": None, "issues": [],
              "source": "<RUN_DIR>/run_metadata.yaml#run_uuid"}
    if type(run_metadata_bytes) is not bytes:
        _issue(result, "invalid_run_metadata", "run_metadata", "Expected raw bytes")
        return result
    try:
        text = run_metadata_bytes.decode("utf-8", errors="strict")
    except UnicodeError:
        _issue(result, "invalid_run_metadata", "run_metadata", "run_metadata is not UTF-8")
        return result
    candidates = [line for line in text.splitlines() if line.startswith("run_uuid:")]
    if len(candidates) != 1:
        _issue(result, "run_uuid_unbound", "run_metadata",
               f"Expected exactly one top-level run_uuid line, found {len(candidates)}")
        return result
    match = _RUN_UUID_LINE.match(candidates[0])
    if match is None:
        _issue(result, "run_uuid_unbound", "run_metadata",
               "run_uuid is not a canonical lowercase UUID")
        return result
    result.update(status="passed", run_uuid=match["uuid"])
    return result


# Rollout outputs (AlpaSim e121e591): worker/main.py:334 rollouts_dir=<log_dir>/rollouts;
# unbound_rollout.py:338 save_path_root=<rollouts_dir>/<scene_id>;
# rollout dir=<save_path_root>/<rollout_uuid> (event_loop.py:252-256), "rollout.asl";
# runtime_evaluator.py:200-202 "metrics.parquet"; autoresume.py:19
# TRACKER_FILE_NAME="_complete" (written after evaluation, event_loop.py:679-681).
_ROLLOUT_FILE = re.compile(
    rf"^rollouts/(?P<scene>{_SCENE})/(?P<uuid>{_ROLLOUT_UUID})/(?P<name>[^/]+)$")
_KNOWN_ROLLOUT_FILES = {"_complete": "complete_marker", "metrics.parquet": "metrics",
                        "rollout.asl": "asl"}


def scan_rollout_files(paths):
    """Group RUN_DIR-relative file paths into rollout folders; pure, no I/O.

    Paths come from a no-follow guest manifest. Files outside ``rollouts/`` are
    ignored; malformed paths under ``rollouts/`` are blocking.
    """
    report = {"folders": [], "issues": []}
    if type(paths) is not list or not all(type(path) is str for path in paths):
        _issue(report, "invalid_rollout_listing", "rollout_files",
               "Expected a list of relative path strings")
        return report
    folders = {}
    for path in sorted(set(paths)):
        if not path.startswith("rollouts/"):
            continue
        match = _ROLLOUT_FILE.match(path)
        if match is None or match["scene"] in (".", ".."):
            _issue(report, "unrecognized_rollout_path", path,
                   "Path under rollouts/ is not <clipgt_id>/<rollout_uuid>/<file>")
            continue
        key = (match["scene"], match["uuid"])
        folder = folders.setdefault(key, {
            "clipgt_id": match["scene"], "rollout_uuid": match["uuid"],
            "complete_marker": False, "metrics": False, "asl": False, "other_files": []})
        flag = _KNOWN_ROLLOUT_FILES.get(match["name"])
        if flag is None:
            folder["other_files"].append(match["name"])
        else:
            folder[flag] = True
    report["folders"] = [folders[key] for key in sorted(folders)]
    return report


# failed_rollouts.py:48-64 builds runtime-failure rows with exactly these metric keys,
# run_uuid None and status "fail"; scored rows (processing.py:290-330) also use
# status "fail" for a scored failure but carry the wide metric columns and run_uuid.
_FAILURE_ROW_METRIC_KEYS = frozenset({"run_name", "run_uuid", "clipgt_id", "rollout_id", "error"})


def _classify_summary_row(row):
    if type(row) is not dict:
        return "malformed"
    metrics = row.get("metrics")
    if (row.get("status") == "fail" and row.get("run_uuid") is None
            and type(metrics) is dict and set(metrics) == _FAILURE_ROW_METRIC_KEYS):
        return "runtime_failure"
    if row.get("status") in ("pass", "fail"):
        return "scored"
    if row.get("status") == "unscored":
        return "unscored"
    return "malformed"


_UNIT_FIELDS = {"scope", "scene_id", "requested_steps", "logs", "rollout_files",
                "run_metadata", "summary", "compose_exit_code"}


def reconcile_unit_attempts(evidence):
    """Reconcile one unit's upstream attempts from logs, folders and summary rows.

    ``evidence`` fields (all required): ``scope`` (one of SCOPES, echoed);
    ``scene_id``; ``requested_steps``;
    ``logs`` (list of ``{source, text}``); ``rollout_files`` (RUN_DIR-relative
    paths); ``run_metadata`` (the UTF-8 text of ``<RUN_DIR>/run_metadata.yaml``
    or ``None``); ``summary`` (parsed ``aggregate/results-summary.json`` or
    ``None``); ``compose_exit_code`` (recorded only).

    An attempt is ``complete`` only with its ``_complete`` marker, its
    ``metrics.parquet``, exactly one non-runtime-failure summary row whose
    clipgt_id/rollout_id/run_uuid match, its Session STARTING and Session
    COMPLETED log lines, and no failure log for it. At least one Compose capture
    is required and every capture must hold runtime-service lines (review
    A4-F1); a pre-UUID failure beside a complete attempt blocks (A4-F3).

    ``status`` uses the shared contract vocabulary: ``passed`` only when exactly
    one complete attempt is the unit's sole attempt; otherwise ``blocked``. The
    domain result is ``unit_outcome``: ``complete``, ``failed``, ``incomplete``
    or ``unresolved`` (blocking evidence issues). A failed or incomplete unit
    carries a blocking ``unit_not_complete`` issue so it never reads as a pass;
    its failure remains part of the complete denominator downstream. The Compose
    exit code is never consulted: under the dev preset failed rollouts still
    aggregate and the runtime exits 0 (simulate/__main__.py:231-265, :310;
    dev.yaml:50). More than one upstream attempt blocks the unit
    (``retry_selection_unregistered``): the retry-selection rule is Huy's.
    """
    try:
        input_sha256 = identity_hash(evidence)
    except (ValueError, TypeError, UnicodeError):
        input_sha256 = None
    report = {
        "artifact_kind": "s2_unit_attempt_evidence", "schema_version": 1, "scope": None,
        "producer": {"module": "gate.s2_capture", "function": "reconcile_unit_attempts",
                     "source_sha256": _SOURCE_SHA256},
        "input_sha256": input_sha256,
        "status": "blocked", "unit_outcome": "unresolved", "attempts": [], "pre_uuid_failures": [], "internal_retries": [],
        "run_uuid_binding": None, "selected_attempt": None, "issues": [],
        "compose_exit_code": {"value": None, "used_for_success": False},
        "runtime_qualified": False, "score_admitted": False,
    }
    if type(evidence) is not dict or set(evidence) != _UNIT_FIELDS:
        missing = sorted(_UNIT_FIELDS - set(evidence)) if type(evidence) is dict else []
        _issue(report, "invalid_unit_evidence", "evidence",
               f"Provide exactly the unit evidence fields; missing {missing}")
        return report
    if type(evidence["scope"]) is not str or evidence["scope"] not in SCOPES:
        _issue(report, "invalid_unit_evidence", "scope",
               f"Declare the evidence scope as one of {list(SCOPES)}")
        return report
    report["scope"] = evidence["scope"]
    scene_id, requested = evidence["scene_id"], evidence["requested_steps"]
    if type(scene_id) is not str or re.fullmatch(_SCENE, scene_id) is None:
        _issue(report, "invalid_unit_evidence", "scene_id", "Scene ID must be a portable label")
        return report
    if type(requested) is not int or requested <= 0:
        _issue(report, "invalid_unit_evidence", "requested_steps", "Positive integer required")
        return report
    exit_code = evidence["compose_exit_code"]
    if exit_code is not None and type(exit_code) is not int:
        _issue(report, "invalid_unit_evidence", "compose_exit_code", "Integer or null")
        return report
    report["compose_exit_code"]["value"] = exit_code

    metadata = evidence["run_metadata"]
    if type(metadata) is str:
        try:
            binding = bind_run_uuid(metadata.encode("utf-8"))
        except UnicodeError:
            binding = {"status": "blocked", "run_uuid": None, "issues": [{
                "code": "invalid_run_metadata", "path": "run_metadata", "severity": "blocking",
                "evidence_refs": [], "message": "run_metadata text is not encodable as UTF-8"}],
                "source": "<RUN_DIR>/run_metadata.yaml#run_uuid"}
    else:
        binding = {"status": "blocked", "run_uuid": None, "issues": [{
            "code": "run_metadata_missing", "path": "run_metadata", "severity": "blocking",
            "evidence_refs": [], "message": "run_metadata.yaml bytes are required"}],
            "source": "<RUN_DIR>/run_metadata.yaml#run_uuid"}
    report["run_uuid_binding"] = binding
    for item in binding["issues"]:
        report["issues"].append(item)
    run_uuid = binding["run_uuid"]

    attempts = {}

    def attempt(uuid):
        return attempts.setdefault(uuid, {
            "rollout_uuid": uuid, "clipgt_ids": [], "started_logged": False,
            "effective_steps": [], "completed_logged": False, "failed_logged": False,
            "retryable": None, "failed_job_ids": [], "folder": None, "summary_rows": [],
            "state": None, "evidence_refs": []})

    retries, pre_uuid = {}, {}
    compose_sources = 0
    logs = evidence["logs"]
    if type(logs) is not list or not logs:
        _issue(report, "runtime_logs_missing", "logs",
               "At least one captured runtime log is required (Compose stdout first)")
        logs = []
    for index, log in enumerate(logs):
        if type(log) is not dict or set(log) != {"source", "text"}:
            _issue(report, "invalid_unit_evidence", f"logs[{index}]", "Expected source and text")
            continue
        parsed = parse_runtime_log(log["text"], source=log["source"])
        for item in parsed["issues"]:
            _issue(report, item["code"], f"logs[{index}].{item['path']}", item["message"])
        if log["source"] in ("compose_up_stdout", "compose_logs"):
            compose_sources += 1
        if log["source"] in _LOG_SOURCES and not parsed["runtime_service_lines"]:
            _issue(report, "runtime_log_without_runtime_lines", f"logs[{index}]",
                   "Capture holds no line attributable to the runtime service; an empty or "
                   "unparseable capture is absence of evidence, never a pass")
        for event in parsed["events"]:
            ref = f"logs[{index}]:line:{event['line']}"
            # The same line may arrive from several captures (Compose stdout and
            # `compose logs`): key by job identity so one failure counts once.
            if event["kind"] == "retry_scheduled":
                retry = retries.setdefault(event["retry_job"], {**event, "evidence_refs": []})
                retry["evidence_refs"].append(ref)
                continue
            if event["kind"] == "rollout_failed" and event["uuid"] == "N/A":
                failure = pre_uuid.setdefault(event["job"], {**event, "evidence_refs": []})
                failure["evidence_refs"].append(ref)
                continue
            current = attempt(event["uuid"])
            current["evidence_refs"].append(ref)
            if event["scene"] not in current["clipgt_ids"]:
                current["clipgt_ids"].append(event["scene"])
            if event["kind"] == "session_started":
                current["started_logged"] = True
                if event["steps"] not in current["effective_steps"]:
                    current["effective_steps"].append(event["steps"])
            elif event["kind"] == "session_completed":
                current["completed_logged"] = True
            else:
                current["failed_logged"] = True
                current["retryable"] = event["retryable"]
                if event["job"] not in current["failed_job_ids"]:
                    current["failed_job_ids"].append(event["job"])

    if logs and not compose_sources:
        _issue(report, "compose_log_missing", "logs",
               "Scheduler retry lines appear only in Compose output (simulate/__main__.py:"
               "296-300); a worker log alone cannot exclude an internal retry")
    for entry in pre_uuid.values():
        del entry["line"]
        report["pre_uuid_failures"].append(entry)
    for entry in retries.values():
        del entry["line"]
        report["internal_retries"].append(entry)

    scan = scan_rollout_files(evidence["rollout_files"])
    for item in scan["issues"]:
        _issue(report, item["code"], f"rollout_files.{item['path']}", item["message"])
    for folder in scan["folders"]:
        current = attempt(folder["rollout_uuid"])
        current["folder"] = folder
        if folder["clipgt_id"] not in current["clipgt_ids"]:
            current["clipgt_ids"].append(folder["clipgt_id"])

    summary = evidence["summary"]
    rows = []
    if summary is not None:
        if type(summary) is not dict or summary.get("schema_version") != 1 or type(
                summary.get("rollouts")) is not list:
            _issue(report, "invalid_summary", "summary",
                   "Expected results-summary.json schema_version 1 with a rollouts list")
        else:
            rows = summary["rollouts"]
    for index, row in enumerate(rows):
        kind = _classify_summary_row(row)
        if kind == "malformed":
            _issue(report, "invalid_summary_row", f"summary.rollouts[{index}]",
                   "Row is neither a scored, unscored nor runtime-failure row")
            continue
        rollout_id = row.get("rollout_id")
        if type(row.get("clipgt_id")) is not str:
            # Pinned writers always set clipgt_id (processing.py scored rows;
            # simulate/__main__.py:58 scenario_id for failure rows).
            _issue(report, "summary_row_scene_missing", f"summary.rollouts[{index}]",
                   "Summary row lacks a string clipgt_id; it cannot be bound to the unit scene")
        entry = {"index": index, "kind": kind, "clipgt_id": row.get("clipgt_id"),
                 "run_uuid": row.get("run_uuid"), "status": row.get("status"),
                 "failure_reason": row.get("failure_reason")}
        if type(rollout_id) is str and re.fullmatch(_ROLLOUT_UUID, rollout_id):
            current = attempt(rollout_id)
            current["summary_rows"].append(entry)
            if type(entry["clipgt_id"]) is str and entry["clipgt_id"] not in current["clipgt_ids"]:
                current["clipgt_ids"].append(entry["clipgt_id"])
        elif kind == "runtime_failure":
            # simulate/__main__.py:58-60: no allocated UUID gives rollout_id "failed-<idx>".
            report["pre_uuid_failures"].append({**entry, "kind": "summary_failure_row",
                                                "rollout_id": rollout_id})
            if type(entry["clipgt_id"]) is str and entry["clipgt_id"] != scene_id:
                _issue(report, "scene_clip_mismatch", f"summary.rollouts[{index}]",
                       f"Failure row clipgt_id {entry['clipgt_id']!r} differs from unit scene")
        else:
            _issue(report, "invalid_summary_row", f"summary.rollouts[{index}]",
                   "Scored row lacks a canonical rollout UUID")

    for uuid in sorted(attempts):
        current = attempts[uuid]
        path = f"attempts[{uuid}]"
        if current["clipgt_ids"] != [scene_id]:
            _issue(report, "scene_clip_mismatch", path,
                   f"Attempt clipgt_id {current['clipgt_ids']} differs from unit scene")
        for steps in current["effective_steps"]:
            if steps > requested:
                _issue(report, "effective_steps_exceed_requested", path,
                       f"Logged steps {steps} exceed requested {requested}")
            elif steps < requested:
                _issue(report, "effective_steps_below_requested", path,
                       f"Logged effective steps {steps} are below requested {requested}",
                       severity="warning")
        if len(current["effective_steps"]) > 1:
            _issue(report, "conflicting_effective_steps", path, "Attempt logs two step counts")
        scored = [row for row in current["summary_rows"] if row["kind"] != "runtime_failure"]
        failures = [row for row in current["summary_rows"] if row["kind"] == "runtime_failure"]
        if len(current["summary_rows"]) > 1:
            _issue(report, "duplicate_summary_rows", path, "One rollout has several summary rows")
        for row in scored:
            if run_uuid is not None and row["run_uuid"] != run_uuid:
                _issue(report, "run_uuid_mismatch", path,
                       "Summary row run_uuid differs from run_metadata.yaml")
        folder = current["folder"] or {}
        failed = current["failed_logged"] or bool(failures)
        if failed and (folder.get("complete_marker") or current["completed_logged"]):
            _issue(report, "contradictory_attempt_evidence", path,
                   "Attempt has both failure and completion evidence")
        if folder and not folder["complete_marker"]:
            _issue(report, "incomplete_rollout_folder", path,
                   "Rollout folder lacks _complete (failed or interrupted attempt)",
                   severity="info")
        if failed:
            current["state"] = "failed"
        elif (folder.get("complete_marker") and folder.get("metrics")
              and len(scored) == 1 and len(current["summary_rows"]) == 1):
            if current["started_logged"] and current["completed_logged"]:
                current["state"] = "complete"
            else:
                current["state"] = "incomplete"
                _issue(report, "attempt_log_evidence_missing", path,
                       "Folder and summary claim completion but the runtime log lacks this "
                       "attempt's Session STARTING and/or Session COMPLETED line")
        else:
            current["state"] = "incomplete"
        report["attempts"].append(current)

    total = len(report["attempts"]) + len(
        [item for item in report["pre_uuid_failures"] if item["kind"] == "rollout_failed"])
    complete = [item for item in report["attempts"] if item["state"] == "complete"]
    if total > 1 or report["internal_retries"]:
        _issue(report, "retry_selection_unregistered", "attempts",
               f"{total} upstream attempts; selecting among them is an owner rule")
    if complete and report["pre_uuid_failures"]:
        _issue(report, "unattributed_failure_with_complete_attempt", "pre_uuid_failures",
               "A runtime failure without a rollout UUID coexists with a complete attempt; "
               "it cannot be attributed, so the unit's attempt count is unresolved")
    if not report["attempts"] and not report["pre_uuid_failures"]:
        _issue(report, "no_upstream_attempt_evidence", "attempts",
               "No log, folder or summary evidence names an attempt")
    if _blocking(report["issues"]):
        report["unit_outcome"] = "unresolved"
    elif len(complete) == 1 and total == 1:
        report["unit_outcome"] = "complete"
        report["status"] = "passed"
        report["selected_attempt"] = complete[0]["rollout_uuid"]
    else:
        failed = any(item["state"] == "failed" for item in report["attempts"]) or bool(
            report["pre_uuid_failures"])
        report["unit_outcome"] = "failed" if failed else "incomplete"
        _issue(report, "unit_not_complete", "unit_outcome",
               f"Unit outcome is {report['unit_outcome']}; it is recorded, never a pass")
    if exit_code == 0 and report["unit_outcome"] != "complete":
        _issue(report, "exit_zero_without_completion", "compose_exit_code",
               "Compose exited 0 without a complete attempt; exit codes are not evidence",
               severity="info")
    return report
