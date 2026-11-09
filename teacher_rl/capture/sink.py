"""Append-only, hash-chained JSONL event sink for in-process S4 capture (H4).

Tickets: S4.3 (capture attempts, retries and dropped payloads before upstream cleanup), E06,
E09, E15 (evidence the ledger consumes), lane-r3 §6 H4. Agent-authored under D-018/D-024.

Each process writes its own stream file ``<label>.jsonl`` under the capture directory, opened
with ``O_CREAT | O_EXCL | O_APPEND | O_NOFOLLOW`` and ``fsync``-ed after every record. Record
``n`` hashes ``{seq, prev, event}`` in canonical-v1 JSON (byte-identical to
``gate.s2_identity.canonical_json_bytes``) and chains to record ``n - 1``; wall and monotonic
times sit outside the hash, so the chain is deterministic. Large payloads go to a
content-addressed blob store (``blobs/<sha[:2]>/<sha>``). Bounds are explicit:
an oversize event is replaced by an ``event_oversize`` record carrying its hash and size; a
blob over budget is recorded by hash and size with ``stored: false``; a stream that would
exceed its byte budget writes ``stream_limit_reached`` and then raises ``SinkLimitError``.
Nothing is dropped silently.

Verification ceiling:
- ``EventSink``/``read_stream``: **L1** (known answers, tamper/truncation/fork negative
  controls in ``tests/test_s4_capture.py``).
- Durability of ``fsync`` on the host filesystem and behaviour under the real worker
  processes: **L3** (``qualify_on_host``).

Stdlib only: this module runs inside the AlpaGym worker environment through PYTHONPATH and
must not import torch, numpy or PhysLoop's ``gate`` package. It never reads credentials,
never touches the network and never deletes a file.
"""

import errno
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
import weakref
from pathlib import Path

STREAM_SCHEMA = "physloop.s4.capture.v1"
GENESIS = "0" * 64
LIMIT_FIELDS = ("max_event_bytes", "max_stream_bytes", "max_blob_bytes", "max_total_blob_bytes")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FINAL_RESERVE = 1024
_LIVE = weakref.WeakSet()


class SinkError(RuntimeError):
    """The sink cannot record evidence; capture must stop (fail closed)."""


class SinkLimitError(SinkError):
    """A declared byte budget would be exceeded."""


def _validate(value, depth=0):
    if depth > 64:
        raise ValueError("event nesting exceeds 64 levels")
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("event numbers must be finite")
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("event keys must be strings")
            _validate(item, depth + 1)
        return
    if kind is list:
        for item in value:
            _validate(item, depth + 1)
        return
    raise ValueError(f"event values must be JSON-native, got {kind.__name__}")


def canonical_bytes(value):
    """Canonical-v1 JSON: strict UTF-8, sorted keys, compact separators, finite numbers."""
    _validate(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _strict_loads(data):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = item
        return result

    def constant(name):
        raise ValueError(f"non-JSON constant {name}")

    return json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=constant)


def validate_limits(limits):
    """Return the four integer byte budgets or raise ``ValueError`` (no defaults)."""
    if type(limits) is not dict or set(limits) != set(LIMIT_FIELDS):
        raise ValueError(f"sink limits must contain exactly {list(LIMIT_FIELDS)}")
    for name in LIMIT_FIELDS:
        if type(limits[name]) is not int or limits[name] <= 0:
            raise ValueError(f"sink limit {name} must be a positive integer")
    if limits["max_event_bytes"] + _FINAL_RESERVE > limits["max_stream_bytes"]:
        raise ValueError("max_stream_bytes must exceed max_event_bytes plus the final reserve")
    if limits["max_event_bytes"] < 2 * _FINAL_RESERVE:
        raise ValueError(f"max_event_bytes must be at least {2 * _FINAL_RESERVE}")
    return dict(limits)


def _ensure_directory(path):
    """Create ``path`` (and parents) as real directories; refuse symlinks."""
    path = Path(path)
    if not path.is_absolute():
        raise SinkError(f"capture directory must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise SinkError(f"capture path component is not a real directory: {current}")
    return path


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


class EventSink:
    """One append-only stream per process; thread-safe; re-opens a child stream after fork."""

    def __init__(self, directory, *, stream_label, limits, wall_ns=time.time_ns,
                 mono_ns=time.monotonic_ns, getpid=os.getpid):
        if type(stream_label) is not str or not _LABEL.fullmatch(stream_label):
            raise SinkError(f"invalid stream label {stream_label!r}")
        try:
            self._limits = validate_limits(limits)
        except ValueError as exc:
            raise SinkError(str(exc)) from exc
        self._directory = _ensure_directory(directory)
        self._wall_ns, self._mono_ns, self._getpid = wall_ns, mono_ns, getpid
        self._lock = threading.Lock()
        self._blob_bytes = 0
        self._label = stream_label
        self._parent = None
        self._open(stream_label)
        _LIVE.add(self)

    @property
    def path(self):
        return self._path

    @property
    def directory(self):
        return self._directory

    def _open(self, label):
        self._path = self._directory / f"{label}.jsonl"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW
        try:
            self._fd = os.open(self._path, flags, 0o600)
        except OSError as exc:
            raise SinkError(f"cannot create stream {self._path}: {exc}") from exc
        self._pid = self._getpid()
        self._seq, self._prev, self._bytes, self._closed, self._limited = 0, GENESIS, 0, False, False
        header = {"schema": STREAM_SCHEMA, "stream_label": label, "pid": self._pid,
                  "limits": dict(self._limits)}
        if self._parent is not None:
            header["parent_stream"] = self._parent
        self._append("stream_opened", header)

    def _reset_after_fork(self):
        self._lock = threading.Lock()

    def _check_process(self):
        pid = self._getpid()
        if pid != self._pid:
            # A forked child must not continue the parent's chain or share its fd.
            self._parent = {"label": self._label, "last_seq": self._seq - 1, "last_hash": self._prev}
            self._label = f"{self._label}-fork{pid}"
            os.close(self._fd)
            self._open(self._label)

    def _record_line(self, kind, payload):
        event = {"kind": kind, **payload}
        body = {"seq": self._seq, "prev": self._prev, "event": event}
        try:
            body_bytes = canonical_bytes(body)
        except ValueError as exc:
            raise SinkError(f"event {kind!r} is not JSON-native: {exc}") from exc
        digest = hashlib.sha256(body_bytes).hexdigest()
        time_info = {"wall_ns": self._wall_ns(), "monotonic_ns": self._mono_ns()}
        line = canonical_bytes({**body, "hash": digest, "time": time_info}) + b"\n"
        return line, digest

    def _append(self, kind, payload):
        line, digest = self._record_line(kind, payload)
        if len(line) > self._limits["max_event_bytes"]:
            event_bytes = canonical_bytes({"kind": kind, **payload})
            line, digest = self._record_line("event_oversize", {
                "original_kind": kind, "bytes": len(event_bytes),
                "sha256": hashlib.sha256(event_bytes).hexdigest()})
        budget = self._limits["max_stream_bytes"] - _FINAL_RESERVE
        if self._bytes + len(line) > budget:
            final, final_digest = self._record_line("stream_limit_reached", {
                "rejected_kind": kind, "stream_bytes": self._bytes,
                "max_stream_bytes": self._limits["max_stream_bytes"]})
            self._commit(final, final_digest)
            self._limited = True
            raise SinkLimitError(f"stream {self._path.name} reached its byte budget")
        self._commit(line, digest)
        return self._seq - 1

    def _commit(self, line, digest):
        try:
            _write_all(self._fd, line)
            os.fsync(self._fd)
        except OSError as exc:
            raise SinkError(f"cannot persist capture record: {exc}") from exc
        self._bytes += len(line)
        self._seq += 1
        self._prev = digest

    def emit(self, kind, payload):
        """Append one event; returns its sequence number. Raises ``SinkError`` (fail closed)."""
        if type(kind) is not str or not _LABEL.fullmatch(kind):
            raise SinkError(f"invalid event kind {kind!r}")
        if type(payload) is not dict or "kind" in payload:
            raise SinkError("event payload must be an object without a 'kind' key")
        with self._lock:
            self._check_process()
            if self._closed or self._limited:
                raise SinkError(f"stream {self._path.name} is closed")
            return self._append(kind, payload)

    def put_blob(self, data):
        """Store bytes content-addressed; over-budget blobs are recorded by hash only."""
        if type(data) is not bytes:
            raise SinkError("blob data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        info = {"sha256": digest, "bytes": len(data)}
        with self._lock:
            self._check_process()
            relative = f"blobs/{digest[:2]}/{digest}"
            target = self._directory / "blobs" / digest[:2] / digest
            if os.path.lexists(target):  # an identical blob costs no budget
                existing = _read_regular(target, len(data) + 1)
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise SinkError(f"blob {relative} exists with different bytes")
                return {**info, "stored": True, "path": relative, "deduplicated": True}
            if (len(data) > self._limits["max_blob_bytes"]
                    or self._blob_bytes + len(data) > self._limits["max_total_blob_bytes"]):
                return {**info, "stored": False, "reason": "blob_over_budget"}
            _ensure_directory(target.parent)
            try:
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except OSError as exc:
                raise SinkError(f"cannot create blob {relative}: {exc}") from exc
            try:
                _write_all(fd, data)
                os.fsync(fd)
            except OSError as exc:
                raise SinkError(f"cannot persist blob {relative}: {exc}") from exc
            finally:
                os.close(fd)
            self._blob_bytes += len(data)
            return {**info, "stored": True, "path": relative}

    def close(self, summary=None):
        """Write ``stream_closed`` with counts and close the descriptor (idempotent)."""
        with self._lock:
            if self._closed:
                return
            if not self._limited and self._getpid() == self._pid:
                self._append("stream_closed", {"records": self._seq + 1,
                                               "blob_bytes": self._blob_bytes,
                                               "summary": summary or {}})
            self._closed = True
            os.close(self._fd)


def _after_fork_in_child():
    for sink in list(_LIVE):
        sink._reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _read_regular(path, max_bytes):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SinkError(f"refusing symlink {path}") from exc
        raise SinkError(f"cannot open {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SinkError(f"not a regular file: {path}")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SinkLimitError(f"{path} exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_stream(path, *, max_bytes):
    """Read one stream with no-follow; verify sequence, chain and hashes.

    Returns ``{"status", "stream_label", "events", "issues", "chain_head"}`` (the last verified
    record hash, which binds the whole verified prefix); events carry their ``seq``.
    A missing final newline (torn write), a broken chain, a gap, a hash mismatch or a final
    ``stream_limit_reached`` record (the sink refused evidence) is a blocking issue; records
    after the first broken record are not returned.
    """
    issues = []

    def issue(code, message):
        issues.append({"code": code, "path": str(path), "severity": "blocking",
                       "evidence_refs": [], "message": message})

    data = _read_regular(path, max_bytes)
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif data:
        issue("truncated_tail", "the last record has no terminating newline")
        lines.pop()
    events, prev, label = [], GENESIS, None
    for index, line in enumerate(lines):
        try:
            record = _strict_loads(line)
        except ValueError as exc:
            issue("malformed_record", f"record {index}: {exc}")
            break
        if type(record) is not dict or set(record) != {"seq", "prev", "event", "hash", "time"}:
            issue("malformed_record", f"record {index} has unexpected fields")
            break
        body = {"seq": record["seq"], "prev": record["prev"], "event": record["event"]}
        if type(record["seq"]) is not int or record["seq"] != index:
            issue("sequence_gap", f"record {index} carries seq {record['seq']}")
            break
        if record["prev"] != prev:
            issue("chain_broken", f"record {index} does not chain to its predecessor")
            break
        digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
        if type(record["hash"]) is not str or not _HEX64.fullmatch(record["hash"]) or (
                record["hash"] != digest):
            issue("hash_mismatch", f"record {index} hash does not match its content")
            break
        event = record["event"]
        if type(event) is not dict or type(event.get("kind")) is not str:
            issue("malformed_record", f"record {index} has no event kind")
            break
        if index == 0:
            if event["kind"] != "stream_opened" or event.get("schema") != STREAM_SCHEMA:
                issue("missing_stream_header", "the first record must open the stream")
                break
            label = event.get("stream_label")
        events.append({**event, "seq": record["seq"]})
        prev = digest
    if not events and not issues:
        issue("empty_stream", "the stream holds no records")
    if events and events[-1]["kind"] == "stream_limit_reached":
        # The sink refused an event and capture stopped: the stream is cut, not closed.
        issue("stream_limit_reached",
              f"the stream reached its byte budget and refused a "
              f"{events[-1].get('rejected_kind')!r} event; later evidence is missing")
    closed = bool(events) and events[-1]["kind"] == "stream_closed"
    if events and not closed and not issues:
        issues.append({"code": "stream_not_closed", "path": str(path), "severity": "warning",
                       "evidence_refs": [], "message": "no stream_closed record (process may "
                                                       "have ended abnormally)"})
    blocking = any(item["severity"] == "blocking" for item in issues)
    return {"status": "blocked" if blocking else "passed", "stream_label": label,
            "events": events, "issues": issues, "chain_head": prev if events else None}
