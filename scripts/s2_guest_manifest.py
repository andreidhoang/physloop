"""No-follow guest manifest (relative path, size, sha256) for S2 export and run-dir hashing.

Tickets: build-spec B1 (runbook section 4 step 8; lane R1 P7 "guest no-follow manifest";
readiness CPU-11 "guest-side manifest for remote transfer"). Agent-authored under
D-018/D-024; serves S2.1-S2.7 [C] (G2, G7).

``build_manifest(root, scope=..., symlink_policy=...)`` walks ``root`` through directory
file descriptors opened with ``O_NOFOLLOW`` on every component (including every ancestor
of ``root``), so no entry outside the root is ever opened or hashed. Each regular file is
hashed by streaming 1 MiB chunks from a descriptor (never by path), and the byte count
read must equal the descriptor's size, otherwise ``file_changed_during_hash`` blocks.
File receipts are exactly ``{path, sha256, size_bytes}`` so ``gate.s2_pilot.verify_export``
and ``gate.s2_identity.read_artifact`` consume them unchanged.

Policies:

- ``symlink_policy="reject"`` (export; the default): any symbolic link is a blocking
  ``symlink_rejected`` issue and is never followed.
- ``symlink_policy="record"`` (run-directory hashing before teardown; AlpaSim writes
  relative links under ``aggregate/videos``, lane R2 section 2): links are listed with
  their literal target text, never followed; a target that is absolute or normalizes
  outside the root is a blocking ``symlink_outside_root`` issue.
- Sockets, FIFOs and devices are blocking ``special_file`` issues. Names that are not
  strict UTF-8 or contain control characters are ``unportable_name``; two paths equal
  after NFC normalization and case folding are ``duplicate_normalized_path`` (shared
  contract rule 2.4). Empty directories are not represented.

Output is deterministic: entries sorted by path, canonical JSON via
``gate.s2_identity``; no wall-clock value is recorded. ``write_manifest`` creates a new
file (``O_EXCL``) and refuses an output path inside the manifested root, so a manifest
never describes itself.

Verification level: ``build_manifest``/``write_manifest``/``main`` L1 on local temporary
trees (CPU known answers, symlink and special-file negative controls). Behaviour on the
guest filesystem (overlay/containerd stores, world-writable container outputs, sizes of
real ASLs) is L3 (``qualify_on_host``).

This module never follows a symbolic link, never reads outside ``root``, never copies,
transfers, deletes or seals anything, never contacts a network and never reads secrets
or environment variables.
"""

import argparse
import hashlib
import os
import posixpath
import stat
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import _directory_fd, canonical_json_bytes, identity_hash

ARTIFACT_KIND = "s2_guest_manifest"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
SYMLINK_POLICIES = ("reject", "record")
CHUNK_BYTES = 1024 * 1024
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _portable(name):
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return not any(ord(char) < 32 or ord(char) == 127 for char in name)


def _hash_descriptor(descriptor, expected_size):
    hasher, count = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb", closefd=False) as stream:
        while True:
            chunk = stream.read(min(CHUNK_BYTES, expected_size + 1 - count))
            if not chunk:
                break
            hasher.update(chunk)
            count += len(chunk)
            if count > expected_size:
                break
    return hasher.hexdigest(), count


def _walk(directory, prefix, report, policy, depth):
    if depth > 256:
        report["issues"].append(_issue("tree_too_deep", prefix or ".", "Nesting exceeds 256"))
        return
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError as exc:
        report["issues"].append(_issue("unreadable", prefix or ".", type(exc).__name__))
        return
    for entry in entries:
        name = entry.name
        relative = f"{prefix}/{name}" if prefix else name
        if not _portable(name):
            report["issues"].append(_issue(
                "unportable_name", relative.encode("utf-8", "backslashreplace").decode(),
                "Names must be strict UTF-8 without control characters"))
            continue
        try:
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except OSError as exc:
            report["issues"].append(_issue("unreadable", relative, type(exc).__name__))
            continue
        mode = metadata.st_mode
        if stat.S_ISLNK(mode):
            if policy == "reject":
                report["issues"].append(_issue(
                    "symlink_rejected", relative, "Symbolic links are never exported or followed"))
                continue
            try:
                target = os.readlink(name, dir_fd=directory)
            except OSError as exc:
                report["issues"].append(_issue("unreadable", relative, type(exc).__name__))
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
            if (not _portable(target) or target.startswith("/") or resolved == ".."
                    or resolved.startswith("../")):
                report["issues"].append(_issue(
                    "symlink_outside_root", relative,
                    "Link target is absolute or leaves the manifested root (not followed)"))
            report["symlinks"].append({"path": relative, "target": target})
        elif stat.S_ISDIR(mode):
            try:
                child = os.open(name, _DIR_FLAGS, dir_fd=directory)
            except OSError as exc:
                report["issues"].append(_issue("unreadable", relative, type(exc).__name__))
                continue
            try:
                _walk(child, relative, report, policy, depth + 1)
            finally:
                os.close(child)
        elif stat.S_ISREG(mode):
            try:
                descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory)
            except OSError as exc:
                report["issues"].append(_issue("unreadable", relative, type(exc).__name__))
                continue
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                        metadata.st_dev, metadata.st_ino):
                    report["issues"].append(_issue(
                        "file_changed_during_hash", relative, "Entry was replaced while opening"))
                    continue
                digest, count = _hash_descriptor(descriptor, opened.st_size)
            except OSError as exc:
                report["issues"].append(_issue("unreadable", relative, type(exc).__name__))
                continue
            finally:
                os.close(descriptor)
            if count != opened.st_size:
                report["issues"].append(_issue(
                    "file_changed_during_hash", relative,
                    f"Read {count} bytes from a file whose size was {opened.st_size}"))
                continue
            report["files"].append({"path": relative, "sha256": digest,
                                    "size_bytes": opened.st_size})
        else:
            report["issues"].append(_issue(
                "special_file", relative, "Only regular files and directories are manifested"))


def duplicate_normalized_issues(paths):
    """Blocking issues for paths equal after NFC normalization and case folding (L1)."""
    issues, seen = [], {}
    for path in sorted(paths):
        key = unicodedata.normalize("NFC", path).casefold()
        if key in seen:
            issues.append(_issue("duplicate_normalized_path", path,
                                 f"Equal after NFC normalization and case folding to "
                                 f"{seen[key]!r}"))
        seen.setdefault(key, path)
    return issues


def build_manifest(root, *, scope, symlink_policy="reject"):
    """Return the deterministic no-follow manifest report for ``root`` (L1)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; no default is chosen")
    if symlink_policy not in SYMLINK_POLICIES:
        raise ValueError(f"symlink_policy must be one of {SYMLINK_POLICIES}")
    absolute = Path(root).absolute()
    report = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": {"module": "scripts.s2_guest_manifest", "function": "build_manifest",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {}, "root": str(absolute), "symlink_policy": symlink_policy,
        "status": "blocked", "issues": [], "files": [], "symlinks": [],
        "units": {"size_bytes": "bytes"}, "totals": None, "files_sha256": None,
    }
    try:
        directory = _directory_fd(absolute)
    except (OSError, ValueError) as exc:
        report["issues"].append(_issue(
            "root_unopenable", ".",
            f"Root must be an existing directory reached without symlinks ({type(exc).__name__})"))
        return report
    try:
        _walk(directory, "", report, symlink_policy, 0)
    finally:
        os.close(directory)
    report["issues"].extend(duplicate_normalized_issues(
        [entry["path"] for entry in report["files"] + report["symlinks"]]))
    report["files"].sort(key=lambda item: item["path"])
    report["symlinks"].sort(key=lambda item: item["path"])
    report["totals"] = {"files": len(report["files"]), "symlinks": len(report["symlinks"]),
                        "size_bytes": sum(item["size_bytes"] for item in report["files"])}
    report["files_sha256"] = identity_hash(report["files"])
    if not any(item["severity"] == "blocking" for item in report["issues"]):
        report["status"] = "passed"
    return report


def write_manifest(report, output, *, root):
    """Write canonical JSON to a new file outside ``root``; never overwrite (L1)."""
    output = Path(output).absolute()
    root = Path(root).absolute()
    if output == root or root in output.parents:
        raise ValueError("The manifest output must lie outside the manifested root")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--output", required=True, help="new file outside ROOT")
    parser.add_argument("--record-symlinks", action="store_true",
                        help="record link targets instead of rejecting links (run-dir hashing)")
    args = parser.parse_args(argv)
    report = build_manifest(args.root, scope=args.scope,
                            symlink_policy="record" if args.record_symlinks else "reject")
    write_manifest(report, args.output, root=args.root)
    print(f"{report['status']}: {report['totals']['files'] if report['totals'] else 0} files, "
          f"{len(report['issues'])} issues -> {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
