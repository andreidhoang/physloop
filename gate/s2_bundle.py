"""S2 evidence sealing, verification and fresh-directory replay (CPU-11).

Tickets: readiness CPU-11 (evidence sealing, safe export and replay); build-spec B2.
Serves S2.1/S2.3/S2.7 [C] (G2, G7) under D-018/D-024. Agent-authored evaluator
engineering; not Huy's mastery evidence.

A bundle is a directory whose regular files are all declared in a caller-supplied
artifact index (``s2_bundle_index``). ``seal_bundle`` hashes every declared file
through descriptor-relative, no-follow reads, scans each one for credential canaries,
checks restriction classes, then writes ``bundle-manifest.json`` with exclusive
creation and, only after every file and the manifest are flushed, the final marker
``BUNDLE_SEALED.json``. The manifest lists every file except itself and the marker
(relative path, size, sha256, role, restriction class, licensed lineage). A bundle
interrupted between the two writes is visibly ``unsealed``: ``verify_bundle`` blocks
it and ``seal_bundle`` refuses to seal it again (no silent repair of evidence).

A same-origin hash manifest detects changed, missing, truncated or added bytes. It is
not a signature and does not prove an honest producer (``authenticity:
not_verified``).

Restriction classes (closed set): ``public_safe`` (code, synthetic or public-source
fixtures without licensed lineage), ``internal`` (actual non-licensed evidence such
as provider receipts), ``internal_kc_l`` (PAI-AV/NuRec-derived outputs and numbers,
internal under KC-L, P-004, licence sections 1 and 3) and ``internal_kc_l2``
(derived weights or labels, internal under KC-L2, licence section 4.6). An entry
with licensed lineage must be ``internal_kc_l`` or ``internal_kc_l2``; an ``actual``
bundle never carries ``public_safe`` entries. ``publication_authorized`` is always
False: publication remains KC-L/KC-L2 and an explicit owner action.

Secret canary scan: environment-assignment names matched case-insensitively (the
credential names the Vast host probe recorded in ``tests/test_vast_s2_host_probe.py``
plus HF/NGC/Vast/GitHub/AWS token names and the bare ``api_key`` the watchdog's
provider rows carry), credential-bearing CLI flags (``--token``, ``--api-key`` and the
``vastai set api-key <key>`` argument form), ``Bearer`` credentials and value shapes
(``gate.s2_runtime.SECRET_PATTERNS``: HF, NGC, GitHub tokens and private keys, plus
AWS access key IDs). The manifest bytes are scanned before they are written (replay
kwargs are copied into it) and again on verification. Findings name the pattern and
byte offset, never the value. A directory the walk cannot list blocks
(``unreadable_directory``): an unlisted directory is never treated as empty.

Bundle minimum (readiness spec CPU-11): an ``actual``-scope bundle must carry every
role in ``ACTUAL_MINIMUM_ROLES`` (registration, source snapshot, runtime inventory,
schedule, attempts, raw-artifact index, extraction, population, diagnostics,
telemetry, cost, effective configuration, lifecycle, export receipt and review), at
seal and at verification. Synthetic and fixture bundles declare their own
``required_roles``.

Replay: ``replay_bundle`` copies the verified bytes into a fresh directory,
re-verifies them there and re-runs each declared deterministic analysis from its
bundled JSON inputs, comparing ``identity_hash`` of the recomputed output with the
bundled output. Producer source hashes are part of the compared content, so replay
is meaningful against the same source snapshot. When the bundle carries an
``s2_source_manifest`` (role ``source_snapshot``) every recorded file is re-hashed
under ``source_root`` (default: this checkout) and a changed or missing file blocks
(``source_drift``); git dirty state is not recorded here. Replay uses the current
interpreter and installed dependencies (``dependency_installation_tested: False``).
Stochastic power replay is exact only under the recorded seed and the same
environment.

Verification level: every function here is L1 (CPU, fixtures and negative
controls). Agent technical defaults (D-018): ``CHUNK_BYTES`` = 1 MiB streaming reads;
``MAX_JSON_BYTES`` = 256 MiB per replay input or output.

This module never uploads, publishes, deletes or rewrites evidence, never follows a
symlink, never repairs a failed bundle and never echoes a matched secret value.
"""

import errno
import hashlib
import os
import re
import stat
import unicodedata
from pathlib import Path

from gate.s2_bootstrap import SECRET_NAMES as BOOTSTRAP_SECRET_NAMES
from gate.s2_identity import _directory_fd, canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_runtime import SECRET_PATTERNS

SCHEMA_VERSION = 1
INDEX_KIND = "s2_bundle_index"
MANIFEST_KIND = "s2_bundle_manifest"
SEAL_KIND = "s2_bundle_seal"
MANIFEST_NAME = "bundle-manifest.json"
SEAL_NAME = "BUNDLE_SEALED.json"
RESERVED_NAMES = (MANIFEST_NAME, SEAL_NAME)
SCOPES = ("synthetic", "public_source_fixture", "actual")
RESTRICTION_CLASSES = ("public_safe", "internal", "internal_kc_l", "internal_kc_l2")
LICENSED_CLASSES = ("internal_kc_l", "internal_kc_l2")
ROLES = (
    "registration", "source_snapshot", "runtime_inventory", "schedule", "attempts",
    "raw_artifact", "raw_artifact_index", "extraction", "population", "diagnostics",
    "telemetry", "cost", "effective_config", "lifecycle", "export_receipt", "review",
    "analysis_input", "analysis_output", "report", "other",
)
# Readiness spec CPU-11 "Bundle minimum", enforced for actual scope.
ACTUAL_MINIMUM_ROLES = (
    "registration", "source_snapshot", "runtime_inventory", "schedule", "attempts",
    "raw_artifact_index", "extraction", "population", "diagnostics", "telemetry", "cost",
    "effective_config", "lifecycle", "export_receipt", "review",
)
SOURCE_MANIFEST_KIND = "s2_source_manifest"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024  # agent technical default (D-018)
MAX_JSON_BYTES = 256 * 1024 * 1024  # agent technical default (D-018)
_OVERLAP_BYTES = 4096
_INDEX_FIELDS = {"artifact_kind", "schema_version", "scope", "bundle_id", "required_roles",
                 "entries"}
_ENTRY_FIELDS = {"path", "role", "restriction_class", "licensed_lineage", "replay"}
_REPLAY_FIELDS = {"function", "inputs", "kwargs"}
_MANIFEST_FIELDS = {"artifact_kind", "schema_version", "scope", "bundle_id", "index_sha256",
                    "entries", "publication_class", "publication_authorized", "authenticity",
                    "units"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256", "role", "restriction_class",
                          "licensed_lineage", "replay"}
_SEAL_FIELDS = {"artifact_kind", "schema_version", "manifest_name", "manifest_sha256",
                "manifest_size_bytes"}
_BUNDLE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNITS = {"size_bytes": "bytes"}

# Credential environment names. The first eight are the names the Vast host probe
# observed in a real login session (tests/test_vast_s2_host_probe.py CREDENTIALS and
# PAM_ADDED); the rest are the token names this project's tools could meet.
SECRET_ENV_NAMES = (
    "CONTAINER_API_KEY", "JUPYTER_TOKEN", "OPEN_BUTTON_TOKEN", "SSH_CLIENT", "SSH_CONNECTION",
    "PUBLIC_KEY", "SSH_PUBLIC_KEY", "PAM_ADDED_SECRET",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "NGC_API_KEY", "NGC_CLI_API_KEY",
    "NVIDIA_API_KEY", "VAST_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "API_KEY",
)
_NAME_ALTERNATION = "|".join(sorted(SECRET_ENV_NAMES, key=len, reverse=True))
# Secret-slot names a compiled bootstrap plan uses to reference a run-time secret file
# (gate.s2_bootstrap.SECRET_NAMES); mapping an environment name to a slot is a reference.
SECRET_REFERENCE_NAMES = BOOTSTRAP_SECRET_NAMES
_REFERENCE_ALTERNATION = "|".join(SECRET_REFERENCE_NAMES)
# An assignment of a non-empty, non-null value to a credential name in any letter case
# (shell, env file, JSON or YAML). A bare mention of the name (an allowlist, a
# docstring), a nested object (a schema) or a mapping to a secret-slot name is not a
# finding.
SECRET_SCAN_PATTERNS = (
    ("credential_env_assignment", re.compile(
        rb"(?<![A-Za-z0-9_])(?i:" + _NAME_ALTERNATION.encode() + rb")(?![A-Za-z0-9_])"
        rb"[\"']?[ \t]*[:=][ \t]*[\"']?(?!null\b|None\b|[\"',{}\s]|(?:"
        + _REFERENCE_ALTERNATION.encode() + rb")[\"'])\S")),
    ("credential_cli_flag", re.compile(
        rb"--(?:hf-token|token|api-key|password)(?:=|[ \t]+)(?![\"']?--)[^\s\"',\]]")),
    # "vastai set api-key <key>": a positional key after the api-key word.
    ("api_key_argument", re.compile(
        rb"(?<![A-Za-z0-9_-])(?i:api[-_]key)[ \t]+[\"']?(?=[A-Za-z0-9_-]*[0-9])"
        rb"[A-Za-z0-9_-]{16,}")),
    # "Authorization: Bearer <credential>" and bare bearer credentials.
    ("bearer_token", re.compile(
        rb"(?<![A-Za-z0-9_])(?i:bearer)[ \t]+(?=[A-Za-z0-9._~+/=-]*[0-9])"
        rb"[A-Za-z0-9._~+/-]{16,}")),
    ("aws_access_key_id", re.compile(rb"AKIA[0-9A-Z]{16}")),
    *((f"token_shape_{index}", re.compile(pattern.pattern.encode()))
      for index, pattern in enumerate(SECRET_PATTERNS)),
)

# Deterministic analyses that replay may re-run: name -> (module, function, allowed
# keyword names). Inputs are bundled JSON files; kwargs are JSON values.
REPLAY_FUNCTIONS = {
    "reconcile_attempts": ("gate.s2_ledger", "reconcile_attempts", {"schedule", "events"}),
    "describe_validity": ("gate.s2_diagnostics", "describe_validity",
                          {"ledger_report", "observations", "records", "validity_definition",
                           "scope"}),
    "audit_population": ("gate.s2_population", "audit_population",
                         {"manifest", "roles", "overlap_rule"}),
    "reconcile_cost": ("gate.s2_cost", "reconcile_cost", {"events", "quote", "scope"}),
    "run_sensitivity": ("gate.s2_power", "run_sensitivity",
                        {"design", "scenarios", "n_sim", "seed"}),
    "summarize_telemetry": ("gate.s2_telemetry", "summarize_telemetry",
                            {"samples", "allocation", "rollouts", "clock_metadata",
                             "max_sample_gap_us", "scope", "stabilization_rule"}),
    "describe_repeats": ("gate.s2_probes", "describe_repeats",
                         {"observations", "method", "scope", "paired_policies"}),
    "evaluate_registered": ("gate.s2_gate", "evaluate_registered",
                            {"records", "registration", "scope", "adapter_metric"}),
    "next_unit_check": ("gate.s2_pilot", "next_unit_check", {"request"}),
    "build_pilot_packet": ("gate.s2_packet", "build_pilot_packet", {"request"}),
    "validate_runtime_inventory": ("gate.s2_runtime", "validate_runtime_inventory",
                                   {"inventory"}),
    "validate_s2_preflight": ("gate.s2_preflight", "validate_s2_preflight", {"manifest"}),
}


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _producer(function):
    return {"module": "gate.s2_bundle", "function": function, "source_sha256": _SOURCE_SHA256}


def _issue(issues, code, path, message, severity="blocking", evidence_refs=()):
    issues.append({"code": code, "path": path, "severity": severity,
                   "evidence_refs": list(evidence_refs), "message": message})


def _blocking(issues):
    return any(item["severity"] == "blocking" for item in issues)


def _safe_relative(value):
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _normalized(name):
    """Collision key: NFC plus case folding (case-insensitive or normalizing filesystems)."""
    return unicodedata.normalize("NFC", name).casefold()


def scan_bytes_for_secrets(chunks):
    """Return sorted ``(pattern_id, first_offset)`` findings over an iterable of chunks.

    Pure over the supplied bytes; a match that straddles a chunk boundary is found
    because each scan includes the previous chunk's tail. Values are never returned.
    """
    findings = {}
    offset = 0
    tail = b""
    for chunk in chunks:
        window = tail + chunk
        base = offset - len(tail)
        for pattern_id, pattern in SECRET_SCAN_PATTERNS:
            if pattern_id in findings:
                continue
            match = pattern.search(window)
            if match is not None:
                findings[pattern_id] = base + match.start()
        offset += len(chunk)
        tail = window[-_OVERLAP_BYTES:]
    return sorted(findings.items())


def _open_relative(root, relative, *, write=False):
    """Descriptor for ``root/relative`` opened without following any symlink."""
    directory = _directory_fd(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parts = relative.split("/")
    try:
        for part in parts[:-1]:
            following = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = following
        if write:
            return os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                           | os.O_CLOEXEC, 0o644, dir_fd=directory)
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                       dir_fd=directory)
    finally:
        os.close(directory)


def _stream(root, relative, *, sync=False, sink=None):
    """Stream one regular file: return ``(size, sha256, secret findings)``.

    ``sync`` fsyncs the file before reading (seal: required artifacts are flushed
    before the marker). ``sink`` receives every chunk (replay copy). Raises OSError or
    ValueError on a missing, special or changing file.
    """
    descriptor = _open_relative(root, relative)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file")
        if sync:
            os.fsync(descriptor)
        digest = hashlib.sha256()
        size = 0

        def chunks():
            nonlocal size
            while True:
                chunk = os.read(descriptor, CHUNK_BYTES)
                if not chunk:
                    return
                size += len(chunk)
                digest.update(chunk)
                if sink is not None:
                    sink(chunk)
                yield chunk

        findings = scan_bytes_for_secrets(chunks())
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if size != before.st_size or (after.st_size, after.st_mtime_ns) != (
            before.st_size, before.st_mtime_ns):
        raise ValueError("file changed while being read")
    return size, digest.hexdigest(), findings


def _read_small(root, relative, max_bytes):
    """Read a bounded regular file without following symlinks; None when absent."""
    try:
        descriptor = _open_relative(root, relative)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{relative} is not a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(f"{relative} exceeds the {max_bytes}-byte bound")
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(descriptor, CHUNK_BYTES)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(descriptor)
    if len(data) > max_bytes:
        raise ValueError(f"{relative} grew past the {max_bytes}-byte bound")
    return data


def _write_new(root, relative, data):
    """Exclusive, no-follow creation with fsync of the file and its directory."""
    descriptor = _open_relative(root, relative, write=True)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(Path(root) / relative.rpartition("/")[0] if "/" in relative else root)


def _fsync_directory(path):
    try:
        descriptor = _directory_fd(path)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as exc:  # some filesystems do not support directory fsync
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise
    finally:
        os.close(descriptor)


def _walk(root, issues):
    """Relative POSIX paths of regular files under root; symlinks/special files are issues."""
    files = []
    root_path = Path(root)

    def unreadable(exc):
        # Review B2-02: a directory the walk cannot list is a blocking finding, never
        # silently skipped (its files would be neither indexed nor scanned).
        name = Path(exc.filename) if exc.filename else root_path
        try:
            relative = name.relative_to(root_path).as_posix()
        except ValueError:
            relative = str(name)
        _issue(issues, "unreadable_directory", relative,
               f"Directory could not be listed ({type(exc).__name__}); its contents are "
               "neither indexed nor scanned")

    for current, directories, names in os.walk(root_path, onerror=unreadable,
                                               followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            full = current_path / name
            relative = full.relative_to(root_path).as_posix()
            if stat.S_ISLNK(os.lstat(full).st_mode):
                _issue(issues, "symlink_in_bundle", relative,
                       "Symlinked directory in the bundle; never followed")
        directories[:] = sorted(name for name in directories
                                if not stat.S_ISLNK(os.lstat(current_path / name).st_mode))
        for name in sorted(names):
            full = current_path / name
            relative = full.relative_to(root_path).as_posix()
            mode = os.lstat(full).st_mode
            if stat.S_ISLNK(mode):
                _issue(issues, "symlink_in_bundle", relative,
                       "Symlink in the bundle; never followed")
            elif not stat.S_ISREG(mode):
                _issue(issues, "special_file_in_bundle", relative,
                       "Only regular files may be bundled")
            else:
                files.append(relative)
    seen = {}
    for relative in files:
        key = _normalized(relative)
        if key in seen:
            _issue(issues, "duplicate_normalized_name", relative,
                   f"Collides with {seen[key]!r} after NFC/case folding")
        seen.setdefault(key, relative)
    return sorted(files)


def _check_replay(entry, path, index_paths, issues):
    replay = entry["replay"]
    if replay is None:
        return
    if type(replay) is not dict or set(replay) != _REPLAY_FIELDS:
        _issue(issues, "invalid_replay", path, f"replay is null or exactly {sorted(_REPLAY_FIELDS)}")
        return
    spec = REPLAY_FUNCTIONS.get(replay["function"]) if type(replay["function"]) is str else None
    if spec is None:
        _issue(issues, "unsupported_replay_function", f"{path}.function",
               f"Replay supports only {sorted(REPLAY_FUNCTIONS)}")
        return
    if entry["role"] != "analysis_output":
        _issue(issues, "invalid_replay", f"{path}.role", "Only analysis_output entries replay")
    inputs, kwargs = replay["inputs"], replay["kwargs"]
    if type(inputs) is not dict or type(kwargs) is not dict:
        _issue(issues, "invalid_replay", path, "inputs and kwargs must be objects")
        return
    if set(inputs) & set(kwargs) or not (set(inputs) | set(kwargs)) <= spec[2]:
        _issue(issues, "invalid_replay", path,
               f"{replay['function']} accepts only {sorted(spec[2])}, each once")
    for name, source in sorted(inputs.items()):
        if type(source) is not str or source not in index_paths or source == entry["path"]:
            _issue(issues, "replay_input_not_bundled", f"{path}.inputs.{name}",
                   "Replay inputs must be other entries of the same bundle")


def validate_index(artifact_index):
    """Structural issues of an ``s2_bundle_index`` (L1); an empty list means usable."""
    issues = []
    try:
        identity_hash(artifact_index)
    except (TypeError, ValueError, UnicodeError):
        _issue(issues, "invalid_index", "index", "Index must be strict JSON")
        return issues
    if type(artifact_index) is not dict or set(artifact_index) != _INDEX_FIELDS:
        _issue(issues, "invalid_index", "index", f"Index fields are exactly {sorted(_INDEX_FIELDS)}")
        return issues
    if (artifact_index["artifact_kind"] != INDEX_KIND
            or type(artifact_index["schema_version"]) is not int
            or artifact_index["schema_version"] != SCHEMA_VERSION):
        _issue(issues, "unsupported_index", "index", f"Require {INDEX_KIND} schema 1")
        return issues
    if artifact_index["scope"] not in SCOPES or type(artifact_index["scope"]) is not str:
        _issue(issues, "invalid_scope", "scope", f"Declare scope as one of {list(SCOPES)}")
    if type(artifact_index["bundle_id"]) is not str or not _BUNDLE_ID.fullmatch(
            artifact_index["bundle_id"]):
        _issue(issues, "invalid_bundle_id", "bundle_id", "1-128 of [A-Za-z0-9._-]")
    required = artifact_index["required_roles"]
    if type(required) is not list or not all(role in ROLES for role in required) or len(
            set(required)) != len(required):
        _issue(issues, "invalid_required_roles", "required_roles",
               f"Unique roles from {list(ROLES)}")
        required = []
    if artifact_index["scope"] == "actual":
        required = [*required, *(role for role in ACTUAL_MINIMUM_ROLES if role not in required)]
    entries = artifact_index["entries"]
    if type(entries) is not list or not entries:
        _issue(issues, "invalid_index", "entries", "Index needs a nonempty entry list")
        return issues
    paths, seen = set(), {}
    for position, entry in enumerate(entries):
        path = f"entries[{position}]"
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            _issue(issues, "invalid_entry", path, f"Entry fields are exactly {sorted(_ENTRY_FIELDS)}")
            continue
        name = entry["path"]
        if not _safe_relative(name):
            _issue(issues, "unsafe_path", f"{path}.path",
                   "Relative POSIX path without traversal, empty or backslash components")
            continue
        if name in RESERVED_NAMES:
            _issue(issues, "reserved_name", f"{path}.path", f"{name} is written by the sealer")
        key = _normalized(name)
        if key in seen:
            _issue(issues, "duplicate_normalized_name", f"{path}.path",
                   f"Duplicates {seen[key]!r} after NFC/case folding")
        seen.setdefault(key, name)
        paths.add(name)
        if type(entry["role"]) is not str or entry["role"] not in ROLES:
            _issue(issues, "unknown_role", f"{path}.role", f"Role is one of {list(ROLES)}")
        if type(entry["licensed_lineage"]) is not bool:
            _issue(issues, "unknown_lineage", f"{path}.licensed_lineage",
                   "licensed_lineage must be declared true or false; unknown blocks")
        restriction = entry["restriction_class"]
        if type(restriction) is not str or restriction not in RESTRICTION_CLASSES:
            _issue(issues, "unknown_restriction_class", f"{path}.restriction_class",
                   f"Restriction class is one of {list(RESTRICTION_CLASSES)}")
        elif entry["licensed_lineage"] is True and restriction not in LICENSED_CLASSES:
            _issue(issues, "licensed_lineage_not_restricted", f"{path}.restriction_class",
                   "Licensed-data-derived outputs are internal under KC-L/KC-L2; never "
                   "public_safe or plain internal")
        elif artifact_index["scope"] == "actual" and restriction == "public_safe":
            _issue(issues, "actual_marked_public_safe", f"{path}.restriction_class",
                   "Actual-scope bundles are internal; keep public-safe material separate")
    for position, entry in enumerate(entries):
        if type(entry) is dict and set(entry) == _ENTRY_FIELDS and _safe_relative(entry["path"]):
            _check_replay(entry, f"entries[{position}].replay", paths, issues)
    roles = {entry.get("role") for entry in entries
             if type(entry) is dict and type(entry.get("role")) is str}
    _missing_roles(required, roles, "required_roles", issues)
    return issues


def _missing_roles(required, roles, path, issues):
    for role in required:
        if role not in roles:
            _issue(issues, "missing_required_role", path, f"No entry has required role {role!r}")


def _publication_class(scope, entries):
    if scope == "actual" or any(entry["restriction_class"] != "public_safe" for entry in entries):
        return "internal"
    return "public_safe_candidate"


def _report(kind, function, scope):
    return {"artifact_kind": kind, "schema_version": SCHEMA_VERSION, "scope": scope,
            "producer": _producer(function), "units": {"size_bytes": "bytes"},
            "status": "blocked", "issues": [], "authenticity": "not_verified",
            "publication_authorized": False}


def seal_bundle(root, artifact_index):
    """Hash, scan and seal ``root`` against ``artifact_index``; marker written last (L1).

    Any blocking issue writes nothing. ``sealed`` is True only after the marker exists.
    """
    scope = artifact_index.get("scope") if type(artifact_index) is dict else None
    report = _report("s2_bundle_seal_report", "seal_bundle", scope if scope in SCOPES else None)
    report.update(sealed=False, manifest_sha256=None, manifest=None,
                  input_hashes={"index": None})
    issues = report["issues"]
    issues.extend(validate_index(artifact_index))
    if _blocking(issues):
        return report
    report["input_hashes"]["index"] = identity_hash(artifact_index)
    try:
        _directory_fd_close(root)
    except (OSError, ValueError) as exc:
        _issue(issues, "invalid_root", "root", f"Bundle root unusable: {type(exc).__name__}")
        return report
    for name in RESERVED_NAMES:
        if os.path.lexists(Path(root) / name):
            _issue(issues, "bundle_manifest_exists", name,
                   "A manifest or seal marker already exists (sealed or interrupted bundle); "
                   "never resealed in place")
    if _blocking(issues):
        return report
    on_disk = _walk(root, issues)
    entries = artifact_index["entries"]
    declared = {entry["path"] for entry in entries}
    for relative in on_disk:
        if relative not in declared:
            _issue(issues, "unindexed_file", relative,
                   "File has no declared role or restriction class")
    manifest_entries = []
    for entry in sorted(entries, key=lambda item: item["path"]):
        try:
            size, digest, findings = _stream(root, entry["path"], sync=True)
        except FileNotFoundError:
            _issue(issues, "missing_artifact", entry["path"], "Declared artifact is absent")
            continue
        except (OSError, ValueError) as exc:
            _issue(issues, "unreadable_artifact", entry["path"],
                   f"Could not read without following links: {type(exc).__name__}")
            continue
        for pattern_id, offset in findings:
            _issue(issues, "secret_canary_detected", entry["path"],
                   f"Credential pattern {pattern_id} at byte {offset}; value not echoed")
        manifest_entries.append({"path": entry["path"], "size_bytes": size, "sha256": digest,
                                 "role": entry["role"],
                                 "restriction_class": entry["restriction_class"],
                                 "licensed_lineage": entry["licensed_lineage"],
                                 "replay": entry["replay"]})
    if _blocking(issues):
        return report
    manifest = {
        "artifact_kind": MANIFEST_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "bundle_id": artifact_index["bundle_id"],
        "index_sha256": report["input_hashes"]["index"], "entries": manifest_entries,
        "publication_class": _publication_class(scope, manifest_entries),
        "publication_authorized": False, "authenticity": "not_verified",
        "units": {"size_bytes": "bytes"},
    }
    data = canonical_json_bytes(manifest) + b"\n"
    # Review B2-08: replay kwargs and declared fields are copied into the manifest.
    for pattern_id, offset in scan_bytes_for_secrets([data]):
        _issue(issues, "secret_canary_detected", MANIFEST_NAME,
               f"Credential pattern {pattern_id} at byte {offset}; value not echoed")
    if _blocking(issues):
        return report
    _write_new(root, MANIFEST_NAME, data)
    manifest_sha256 = hashlib.sha256(data).hexdigest()
    seal = {"artifact_kind": SEAL_KIND, "schema_version": SCHEMA_VERSION,
            "manifest_name": MANIFEST_NAME, "manifest_sha256": manifest_sha256,
            "manifest_size_bytes": len(data)}
    _write_new(root, SEAL_NAME, canonical_json_bytes(seal) + b"\n")
    report.update(status="passed", sealed=True, manifest_sha256=manifest_sha256,
                  manifest=manifest)
    return report


def _directory_fd_close(root):
    os.close(_directory_fd(root))


def _validate_manifest(manifest, issues):
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS or (
            manifest["artifact_kind"] != MANIFEST_KIND
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        _issue(issues, "invalid_manifest", MANIFEST_NAME, f"Require {MANIFEST_KIND} schema 1")
        return False
    entries = manifest["entries"]
    if type(entries) is not list or not all(
            type(entry) is dict and set(entry) == _MANIFEST_ENTRY_FIELDS
            and _safe_relative(entry["path"]) and entry["path"] not in RESERVED_NAMES
            and type(entry["size_bytes"]) is int and entry["size_bytes"] >= 0
            and type(entry["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            for entry in entries):
        _issue(issues, "invalid_manifest", f"{MANIFEST_NAME}.entries",
               "Every entry needs a safe path, size_bytes and lowercase sha256")
        return False
    if len({_normalized(entry["path"]) for entry in entries}) != len(entries):
        _issue(issues, "duplicate_normalized_name", f"{MANIFEST_NAME}.entries",
               "Manifest names collide after NFC/case folding")
        return False
    # Review B2-04: a consistent same-origin re-seal cannot widen the closed sets.
    problems = []
    if type(manifest["scope"]) is not str or manifest["scope"] not in SCOPES:
        problems.append(f"scope is one of {list(SCOPES)}")
    if type(manifest["bundle_id"]) is not str or not _BUNDLE_ID.fullmatch(manifest["bundle_id"]):
        problems.append("bundle_id is 1-128 of [A-Za-z0-9._-]")
    if type(manifest["index_sha256"]) is not str or not _SHA256.fullmatch(
            manifest["index_sha256"]):
        problems.append("index_sha256 is 64 lowercase hex")
    if manifest["authenticity"] != "not_verified":
        problems.append("authenticity is always not_verified")
    if manifest["units"] != _UNITS:
        problems.append(f"units are exactly {_UNITS}")
    if type(manifest["publication_class"]) is not str or manifest["publication_class"] not in (
            "internal", "public_safe_candidate"):
        problems.append("publication_class is internal or public_safe_candidate")
    if not all(type(entry["role"]) is str and entry["role"] in ROLES for entry in entries):
        problems.append(f"every role is one of {list(ROLES)}")
    if not all(type(entry["restriction_class"]) is str for entry in entries):
        problems.append("every restriction_class is a string")
    if problems:
        _issue(issues, "invalid_manifest", MANIFEST_NAME, "; ".join(problems))
        return False
    return True


def verify_bundle(root, manifest=None):
    """Verify a sealed bundle's marker, manifest, bytes, secrets and classes (L1).

    ``seal_state`` is ``sealed``, ``unsealed`` (no marker: interrupted or never sealed)
    or ``invalid``. ``verified`` is True only for a sealed bundle whose every byte
    matches; it is never authenticity. A supplied ``manifest`` must equal the bundled one.
    """
    report = _report("s2_bundle_verification", "verify_bundle", None)
    report.update(verified=False, seal_state="invalid", manifest_sha256=None,
                  entries_checked=0, manifest=None)
    issues = report["issues"]
    try:
        _directory_fd_close(root)
        seal_bytes = _read_small(root, SEAL_NAME, 64 * 1024)
        manifest_bytes = _read_small(root, MANIFEST_NAME, MAX_JSON_BYTES)
    except (OSError, ValueError) as exc:
        _issue(issues, "unreadable_bundle", "root", f"Bundle unreadable: {type(exc).__name__}")
        return report
    if seal_bytes is None:
        report["seal_state"] = "unsealed"
        _issue(issues, "bundle_unsealed", SEAL_NAME,
               "No seal marker: the bundle was never sealed or sealing was interrupted")
    if manifest_bytes is None:
        _issue(issues, "missing_manifest", MANIFEST_NAME, "No bundle manifest")
        return report
    report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if seal_bytes is not None:
        try:
            seal = read_json_bytes(seal_bytes)
        except ValueError:
            seal = None
        if type(seal) is not dict or set(seal) != _SEAL_FIELDS or seal.get(
                "artifact_kind") != SEAL_KIND or seal.get("manifest_name") != MANIFEST_NAME or (
                type(seal.get("schema_version")) is not int
                or seal["schema_version"] != SCHEMA_VERSION):
            _issue(issues, "invalid_seal", SEAL_NAME, f"Require a {SEAL_KIND} schema 1 marker")
        elif (seal["manifest_sha256"] != report["manifest_sha256"]
              or seal["manifest_size_bytes"] != len(manifest_bytes)):
            _issue(issues, "manifest_changed_after_seal", MANIFEST_NAME,
                   "Manifest bytes differ from the seal marker")
        else:
            report["seal_state"] = "sealed"
    manifest_secrets = scan_bytes_for_secrets([manifest_bytes, seal_bytes or b""])
    for pattern_id, offset in manifest_secrets:
        _issue(issues, "secret_canary_detected", MANIFEST_NAME,
               f"Credential pattern {pattern_id} at byte {offset} of the manifest/seal; value "
               "not echoed")
    try:
        stored = read_json_bytes(manifest_bytes)
    except ValueError:
        _issue(issues, "invalid_manifest", MANIFEST_NAME, "Manifest is not strict JSON")
        return report
    if not _validate_manifest(stored, issues):
        return report
    report["scope"] = stored["scope"]
    # A manifest carrying a credential is never echoed back.
    report["manifest"] = None if manifest_secrets else stored
    if stored["scope"] == "actual":
        _missing_roles(ACTUAL_MINIMUM_ROLES, {entry["role"] for entry in stored["entries"]},
                       f"{MANIFEST_NAME}.entries", issues)
    if manifest is not None:
        try:
            same = identity_hash(manifest) == identity_hash(stored)
        except (TypeError, ValueError, UnicodeError):
            same = False
        if not same:
            _issue(issues, "manifest_mismatch", MANIFEST_NAME,
                   "The supplied manifest differs from the bundled manifest")
    for entry in stored["entries"]:
        restriction = entry["restriction_class"]
        if restriction not in RESTRICTION_CLASSES or type(entry["licensed_lineage"]) is not bool:
            _issue(issues, "unknown_restriction_class", entry["path"],
                   "Unknown restriction class or lineage")
        elif entry["licensed_lineage"] and restriction not in LICENSED_CLASSES:
            _issue(issues, "licensed_lineage_not_restricted", entry["path"],
                   "Licensed-data-derived entry is not KC-L/KC-L2 internal")
        elif stored["scope"] == "actual" and restriction == "public_safe":
            _issue(issues, "actual_marked_public_safe", entry["path"],
                   "Actual-scope bundles are internal")
    if stored["publication_authorized"] is not False:
        _issue(issues, "publication_claimed", MANIFEST_NAME, "Bundles never authorize publication")
    expected_class = _publication_class(stored["scope"], stored["entries"])
    if stored["publication_class"] != expected_class:
        _issue(issues, "publication_class_mismatch", MANIFEST_NAME,
               f"Entries imply publication_class {expected_class!r}")
    for position, entry in enumerate(stored["entries"]):
        _check_replay(entry, f"{MANIFEST_NAME}.entries[{position}].replay",
                      {item["path"] for item in stored["entries"]}, issues)
    walk_issues = []
    on_disk = set(_walk(root, walk_issues))
    issues.extend(walk_issues)
    listed = {entry["path"] for entry in stored["entries"]}
    for relative in sorted(on_disk - listed - set(RESERVED_NAMES)):
        _issue(issues, "unindexed_file", relative, "File present but not in the manifest")
    for entry in stored["entries"]:
        try:
            size, digest, findings = _stream(root, entry["path"])
        except FileNotFoundError:
            _issue(issues, "missing_artifact", entry["path"], "Manifest entry is absent")
            continue
        except (OSError, ValueError) as exc:
            _issue(issues, "unreadable_artifact", entry["path"],
                   f"Could not read without following links: {type(exc).__name__}")
            continue
        report["entries_checked"] += 1
        if size != entry["size_bytes"]:
            _issue(issues, "size_mismatch", entry["path"],
                   f"{size} bytes on disk, {entry['size_bytes']} in the manifest (truncated or "
                   "extended copy)")
        elif digest != entry["sha256"]:
            _issue(issues, "changed_bytes", entry["path"], "SHA-256 differs from the manifest")
        for pattern_id, offset in findings:
            _issue(issues, "secret_canary_detected", entry["path"],
                   f"Credential pattern {pattern_id} at byte {offset}; value not echoed")
    report["verified"] = report["seal_state"] == "sealed" and not _blocking(issues)
    report["status"] = "passed" if report["verified"] else "blocked"
    return report


def _load_json(root, entry):
    data = _read_small(root, entry["path"], MAX_JSON_BYTES)
    if data is None or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['path']} changed or vanished before replay")
    return read_json_bytes(data)


def _run_analysis(name, arguments):
    import importlib

    module_name, function_name, allowed = REPLAY_FUNCTIONS[name]
    if not set(arguments) <= allowed:
        raise ValueError(f"{name} accepts only {sorted(allowed)}")
    function = getattr(importlib.import_module(module_name), function_name)
    return function(**arguments)


def _compare_source(destination, manifest, source_root, issues):
    """Re-hash every file a bundled ``s2_source_manifest`` records under ``source_root``."""
    snapshots = [entry for entry in manifest["entries"] if entry["role"] == "source_snapshot"]
    result = {"status": "not_recorded", "files_checked": 0, "changed": [], "missing": []}
    for entry in snapshots:
        try:
            recorded = _load_json(destination, entry)
        except ValueError:
            recorded = None
        if type(recorded) is not dict or recorded.get("artifact_kind") != SOURCE_MANIFEST_KIND:
            continue
        files = recorded.get("files")
        if type(files) is not list or not all(
                type(item) is dict and _safe_relative(item.get("path"))
                and type(item.get("sha256")) is str and _SHA256.fullmatch(item["sha256"])
                for item in files):
            result["status"] = "invalid"
            _issue(issues, "invalid_source_manifest", entry["path"],
                   "Source manifest files need safe paths and lowercase sha256")
            continue
        for item in files:
            result["files_checked"] += 1
            try:
                _size, digest, _findings = _stream(source_root, item["path"])
            except FileNotFoundError:
                result["missing"].append(item["path"])
                continue
            except (OSError, ValueError):
                result["missing"].append(item["path"])
                continue
            if digest != item["sha256"]:
                result["changed"].append(item["path"])
        if result["status"] != "invalid":
            result["status"] = "identical"
    result["changed"].sort()
    result["missing"].sort()
    if result["changed"] or result["missing"]:
        result["status"] = "drifted"
        _issue(issues, "source_drift", "source_root",
               f"Replay source differs from the bundled source snapshot: "
               f"{len(result['changed'])} changed, {len(result['missing'])} missing")
    return result


def replay_bundle(root, destination, *, source_root=None):
    """Copy a verified bundle to a fresh ``destination``, re-verify and re-run analyses (L1).

    ``destination`` must not exist (its parent must). ``replay_identical`` is True only
    when the copy verifies, the recorded source snapshot (if any) matches the bytes under
    ``source_root`` (default: this checkout) and every declared analysis reproduces its
    bundled output's identity hash. A mismatch is reported, never repaired.
    """
    report = _report("s2_bundle_replay", "replay_bundle", None)
    report.update(source=None, copy=None, analyses=[], replay_identical=False,
                  source_comparison=None,
                  dependency_installation_tested=False,
                  environment="current interpreter and installed dependencies; fresh-environment "
                              "installation not tested")
    issues = report["issues"]
    source = verify_bundle(root)
    report["source"] = {key: source[key] for key in ("status", "seal_state", "manifest_sha256",
                                                     "entries_checked")}
    report["scope"] = source["scope"]
    if not source["verified"]:
        _issue(issues, "source_bundle_unverified", "root",
               f"Source bundle did not verify: {sorted({i['code'] for i in source['issues']})}")
        return report
    destination = Path(destination)
    try:
        destination.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        _issue(issues, "destination_exists", "destination", "Replay needs a fresh directory")
        return report
    except OSError as exc:
        _issue(issues, "invalid_destination", "destination", type(exc).__name__)
        return report
    manifest = source["manifest"]
    try:
        for entry in manifest["entries"]:
            parent = entry["path"].rpartition("/")[0]
            if parent:
                (destination / parent).mkdir(parents=True, exist_ok=True)
            descriptor = _open_relative(destination, entry["path"], write=True)
            try:
                def sink(chunk, descriptor=descriptor):
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(descriptor, view):]

                size, digest, _findings = _stream(root, entry["path"], sink=sink)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if (size, digest) != (entry["size_bytes"], entry["sha256"]):
                raise ValueError(f"{entry['path']} changed while copying")
        for name in (MANIFEST_NAME, SEAL_NAME):
            _write_new(destination, name, _read_small(root, name, MAX_JSON_BYTES))
    except (OSError, ValueError) as exc:
        _issue(issues, "copy_failed", "destination", f"{type(exc).__name__}: {exc}")
        return report
    copied = verify_bundle(destination, manifest)
    report["copy"] = {key: copied[key] for key in ("status", "seal_state", "manifest_sha256",
                                                   "entries_checked")}
    if not copied["verified"]:
        _issue(issues, "copy_unverified", "destination",
               f"Fresh copy did not verify: {sorted({i['code'] for i in copied['issues']})}")
        return report
    report["source_comparison"] = _compare_source(
        destination, manifest, REPOSITORY_ROOT if source_root is None else Path(source_root),
        issues)
    by_path = {entry["path"]: entry for entry in manifest["entries"]}
    for entry in manifest["entries"]:
        replay = entry["replay"]
        if replay is None:
            continue
        row = {"path": entry["path"], "function": replay["function"],
               "recorded_sha256": None, "replayed_sha256": None, "identical": False}
        report["analyses"].append(row)
        try:
            recorded = _load_json(destination, entry)
            arguments = {name: _load_json(destination, by_path[path])
                         for name, path in sorted(replay["inputs"].items())}
            arguments.update(replay["kwargs"])
            row["recorded_sha256"] = identity_hash(recorded)
            row["replayed_sha256"] = identity_hash(_run_analysis(replay["function"], arguments))
        except Exception as exc:  # noqa: BLE001 - any analysis failure is a replay finding
            _issue(issues, "replay_failed", entry["path"], f"{type(exc).__name__}: {exc}")
            continue
        row["identical"] = row["recorded_sha256"] == row["replayed_sha256"]
        if not row["identical"]:
            _issue(issues, "replay_mismatch", entry["path"],
                   "Recomputed output differs from the bundled output")
    report["replay_identical"] = not _blocking(issues)
    report["status"] = "passed" if report["replay_identical"] else "blocked"
    return report
