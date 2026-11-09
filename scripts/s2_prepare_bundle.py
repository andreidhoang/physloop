#!/usr/bin/env python3
"""Build and verify the sealed PhysLoop source bundle the S2 guest runtime is provisioned from.

Tickets: completeness-audit CPU-09 guest runtime (F3: "scripts/s2_prepare_bundle.py: build a
deterministic source bundle ... with a manifest of relative path, size and sha256 and a bundle
sha256; reject symlinks, absolute paths and paths outside the root; no network"), readiness
CPU-09 relocatable bundle. Agent-authored evaluator engineering under D-018/D-024; serves
S2.1/S2.2/S2.7 [C] (G2, G7). The guest side is ``gate.s2_bootstrap`` stages A6f/A6g (archive
bytes, unpack, ``source_tree`` rule).

Usage (on the operator's machine; nothing is uploaded)::

    python scripts/s2_prepare_bundle.py build --root /abs/physloop --out-dir /abs/new/dir \
        --scope actual
    python scripts/s2_prepare_bundle.py verify /abs/dir/physloop-source-<sha>.tar \
        --bundle-sha256 <sha>

Contents (an explicit allowlist, never "everything not ignored"): ``INCLUDE_FILES``
(``pyproject.toml``, ``uv.lock``, ``.python-version``), every file under ``INCLUDE_DIRS``
(``gate/``, ``scripts/``) and ``EXPERIMENT_FILES`` (the S2 setup draft whose pins the guest
runtime is provisioned from; no guest-side module reads another ``experiments/`` file).
``tests/``, ``teacher_rl/``, ``evidence/`` and ``docs/`` are never included: they are outside
the allowlist, and an include entry under them is refused. Inside the allowlist the files the
repository's ``.gitignore`` ignores are skipped by name (``EXCLUDE_DIR_NAMES``,
``EXCLUDE_FILE_PATTERNS``; ``tests/test_s2_prepare_bundle.py`` checks the list still covers
``.gitignore``); no ``git`` process is run, so uncommitted edits to allowlisted files are
bundled and the manifest, not a commit, is the identity.

Self-reference, stated: ``experiments/S2_SETUP_DRAFT.json`` is bundled and also has
``guest_runtime.source_bundle`` slots. Recording a bundle's identity into the draft would change
the bundled bytes and make that identity stale by construction, so the bundled draft keeps those
slots null; the identity lives only in the run's runtime inventory, which the compiler and runner
read from outside the bundle.

Refusals (the bundle is not written, ``status: "blocked"``): a symlink anywhere in the selection
(file, directory, or any parent directory of an include file below the root), a non-regular file, a path that is absolute, contains ``..``, leaves the
root or is not portable (``gate.s2_runtime.is_source_path``), a missing include, a file whose
text resembles a credential (``gate.s2_runtime.secret_like``; the path is named, never the
value), a path a ustar header cannot hold (``path_too_long``), an output directory that is relative, missing or inside the root, or an output file
that already exists (exclusive creation; nothing is overwritten).

Determinism. The manifest (``s2_source_manifest`` v1, ``gate.s2_runtime.
validate_source_manifest``) lists ``{path, size_bytes, sha256}`` sorted by path and carries no
clock, host path or user name; ``bundle_sha256`` = ``gate.s2_identity.identity_hash`` of it. The
archive is an uncompressed POSIX ustar file: the manifest (``physloop-source-manifest.json``,
canonical JSON bytes) first, then the files in path order, each with mtime 0, uid/gid 0, empty
owner names and mode 0644 (0755 when the source file has an execute bit). Two builds of the
same bytes give byte-identical archives whatever the order or timestamps of the source tree.
The archive name is ``gate.s2_runtime.source_bundle_archive_name(bundle_sha256)``; a copy of
the manifest is written beside it as ``<archive stem>.manifest.json``.

Verification level: ``collect_sources``, ``build_manifest``, ``build_archive`` and
``verify_archive`` are **L1** (tiny synthetic trees, literal known answers and negative
controls); ``build_bundle``/``main`` are L1 file effects confined to the given root and output
directory. Whether the guest's GNU tar unpacks the archive as expected is **L3** (stage A6g).

Never: opens a network connection, runs a process (no ``git``, no ``tar``), follows a
symlink, reads outside the root, overwrites a file, or embeds a credential.
"""

import argparse
import fnmatch
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_runtime import (
    SCOPES,
    SOURCE_MANIFEST_KIND,
    SOURCE_MANIFEST_NAME,
    is_source_path,
    secret_like,
    source_bundle_archive_name,
    validate_source_manifest,
)

SCHEMA_VERSION = 1
INCLUDE_FILES = ("pyproject.toml", "uv.lock", ".python-version")
INCLUDE_DIRS = ("gate", "scripts")
EXPERIMENT_FILES = ("experiments/S2_SETUP_DRAFT.json",)
NEVER_INCLUDED = ("tests", "teacher_rl", "evidence", "docs")
# Mirrors the repository .gitignore entries that can occur inside the allowlist.
EXCLUDE_DIR_NAMES = (
    ".idea", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", ".vscode", "__pycache__",
    "build", "data", "datasets", "dist", "study",
)
EXCLUDE_DIR_PATTERNS = ("*.egg-info",)
EXCLUDE_FILE_PATTERNS = (
    "*.py[cod]", "*.swp", ".DS_Store", ".env", ".env.*", "*.nsys-rep", "nsys-*.sqlite",
    "*.usdz", "*.ply", "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx",
    "*.engine", "*.plan", "*.trt",
)
FILE_MODE, EXEC_MODE = 0o644, 0o755
MAX_FILE_BYTES = 64 * 1024 * 1024  # agent technical default (D-018): per bundled source file
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024  # agent technical default (D-018): whole archive, verify too


def _issue(code, path, message):
    return {"code": code, "path": path, "severity": "blocking", "evidence_refs": [],
            "message": message}


def _producer():
    try:
        source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        source_sha256 = None
    return {"module": "scripts.s2_prepare_bundle", "source_sha256": source_sha256}


def _excluded_dir(name):
    return name in EXCLUDE_DIR_NAMES or any(fnmatch.fnmatchcase(name, p)
                                            for p in EXCLUDE_DIR_PATTERNS)


def _excluded_file(name):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in EXCLUDE_FILE_PATTERNS)


def _read_regular(path, limit=None):
    """Bytes and execute bit of a regular file, opened without following a symlink.

    ``limit`` defaults to ``MAX_FILE_BYTES`` (a bundled source file); ``verify`` passes
    ``MAX_ARCHIVE_BYTES`` for the archive itself.
    """
    limit = MAX_FILE_BYTES if limit is None else limit
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise IsADirectoryError(path)
        if info.st_size > limit:
            raise ValueError("file exceeds the bundle bound")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks), bool(info.st_mode & 0o111)


def _fits_ustar(relative):
    """Whether tarfile's ustar writer can store ``relative`` (name 100 / prefix 155 bytes)."""
    try:
        tarfile.TarInfo(relative).tobuf(tarfile.USTAR_FORMAT)
    except ValueError:
        return False
    return True


def _inside_root(root, full):
    """No component of ``full`` below the (already real) ``root`` is a symlink."""
    return os.path.realpath(full) == full and os.path.commonpath([root, full]) == root


def _check_include(entry):
    parts = entry.split("/") if type(entry) is str else []
    if not is_source_path(entry):
        return "unsafe_path"
    if parts[0] in NEVER_INCLUDED:
        return "excluded_root"
    return None


def collect_sources(root, *, include_files=INCLUDE_FILES, include_dirs=INCLUDE_DIRS,
                    experiment_files=EXPERIMENT_FILES):
    """``(files, excluded, issues)`` for the allowlist under ``root`` (L1; reads files only).

    ``files`` maps relative path -> ``(bytes, executable)``; ``excluded`` lists the
    ``.gitignore``-equivalent paths skipped (sorted). Any issue means the bundle must not be
    built.
    """
    files, excluded, issues = {}, [], []
    root = os.path.realpath(root)

    def add_file(relative, full):
        if not is_source_path(relative):
            issues.append(_issue("unsafe_path", relative, "Not a portable relative path."))
            return
        if not _fits_ustar(relative):
            issues.append(_issue("path_too_long", relative, "Path does not fit a ustar header "
                                 "(name 100, prefix 155 bytes)."))
            return
        if os.path.islink(full) or (os.path.lexists(full) and not _inside_root(root, full)):
            # Final component or any parent directory below the root (F3-3 review).
            issues.append(_issue("symlink_rejected", relative, "Symlinks are never bundled."))
            return
        try:
            data, executable = _read_regular(full)
        except FileNotFoundError:
            issues.append(_issue("missing_input", relative, "Included file does not exist."))
            return
        except (OSError, ValueError) as exc:
            issues.append(_issue("not_regular_file", relative,
                                 f"Only regular files are bundled ({type(exc).__name__})."))
            return
        if secret_like(data.decode("utf-8", errors="replace")):
            issues.append(_issue("secret_like_content", relative,
                                 "File text resembles a credential; it is never bundled."))
            return
        if relative.casefold() in {name.casefold() for name in files}:
            issues.append(_issue("duplicate_path", relative, "Path repeats (case-folded)."))
            return
        files[relative] = (data, executable)

    for entry in (*include_files, *experiment_files):
        problem = _check_include(entry)
        if problem:
            issues.append(_issue(problem, str(entry), "Include entries are portable relative "
                                 "paths outside tests/, teacher_rl/, evidence/ and docs/."))
            continue
        add_file(entry, os.path.join(root, *entry.split("/")))
    for entry in include_dirs:
        problem = _check_include(entry)
        if problem:
            issues.append(_issue(problem, str(entry), "Include entries are portable relative "
                                 "paths outside tests/, teacher_rl/, evidence/ and docs/."))
            continue
        top = os.path.join(root, *entry.split("/"))
        if os.path.islink(top) or not os.path.isdir(top):
            issues.append(_issue("symlink_rejected" if os.path.islink(top) else "missing_input",
                                 entry, "Included directories are real directories."))
            continue

        def unreadable(error):
            raise error

        try:
            for directory, dirnames, filenames in os.walk(top, followlinks=False,
                                                          onerror=unreadable):
                kept = []
                for name in sorted(dirnames):
                    full = os.path.join(directory, name)
                    relative = os.path.relpath(full, root).replace(os.sep, "/")
                    if os.path.islink(full):
                        issues.append(_issue("symlink_rejected", relative,
                                             "Symlinked directories are never bundled."))
                    elif _excluded_dir(name):
                        excluded.append(relative + "/")
                    else:
                        kept.append(name)
                dirnames[:] = kept
                for name in sorted(filenames):
                    full = os.path.join(directory, name)
                    relative = os.path.relpath(full, root).replace(os.sep, "/")
                    if _excluded_file(name) and not os.path.islink(full):
                        excluded.append(relative)
                        continue
                    add_file(relative, full)
        except OSError as exc:
            issues.append(_issue("unreadable_directory", entry,
                                 f"A directory could not be listed ({type(exc).__name__})."))
    return files, sorted(excluded), issues


def build_manifest(files, *, scope):
    """The ``s2_source_manifest`` for collected files (L1; deterministic, no clock)."""
    return {
        "artifact_kind": SOURCE_MANIFEST_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "producer": _producer(),
        "inputs": {"include_files": list(INCLUDE_FILES), "include_dirs": list(INCLUDE_DIRS),
                   "experiment_files": list(EXPERIMENT_FILES),
                   "exclude_dir_names": list(EXCLUDE_DIR_NAMES),
                   "exclude_dir_patterns": list(EXCLUDE_DIR_PATTERNS),
                   "exclude_file_patterns": list(EXCLUDE_FILE_PATTERNS)},
        "units": {"size_bytes": "bytes"},
        "files": [{"path": path, "size_bytes": len(files[path][0]),
                   "sha256": hashlib.sha256(files[path][0]).hexdigest()}
                  for path in sorted(files)],
        "issues": [],
    }


def _member(name, data, mode):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def build_archive(files, manifest):
    """Deterministic ustar bytes: manifest first, then files in path order (L1)."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        data = canonical_json_bytes(manifest)
        archive.addfile(_member(SOURCE_MANIFEST_NAME, data, FILE_MODE), io.BytesIO(data))
        for path in sorted(files):
            data, executable = files[path]
            archive.addfile(_member(path, data, EXEC_MODE if executable else FILE_MODE),
                            io.BytesIO(data))
    return stream.getvalue()


def verify_archive(data, bundle_sha256):
    """Issues for archive bytes against a bundle identity; empty means it verifies (L1).

    Every member must be a regular file at a portable relative path, appear once and match the
    embedded manifest's size and sha256; the manifest must hash to ``bundle_sha256``; nothing
    may be missing or extra. Never raises; never extracts.
    """
    issues = []
    members = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for info in archive.getmembers():
                if not info.isreg() or not (info.name == SOURCE_MANIFEST_NAME
                                            or is_source_path(info.name)):
                    issues.append(_issue("unsafe_member", str(info.name)[:200],
                                         "Members are regular files at portable paths."))
                    continue
                if info.name in members:
                    issues.append(_issue("duplicate_member", info.name, "Member repeats."))
                    continue
                members[info.name] = archive.extractfile(info).read()
    except (tarfile.TarError, OSError, ValueError, TypeError, EOFError) as exc:
        return [_issue("unreadable_archive", "archive", f"Not a tar archive ({type(exc).__name__}).")]
    if issues:
        return issues
    try:
        manifest = read_json_bytes(members.pop(SOURCE_MANIFEST_NAME))
    except (KeyError, ValueError):
        return [_issue("manifest_unreadable", SOURCE_MANIFEST_NAME, "Embedded manifest missing "
                       "or not strict JSON.")]
    problems = validate_source_manifest(manifest)
    if problems:
        return [_issue("manifest_invalid", SOURCE_MANIFEST_NAME,
                       f"Manifest fails validation: {sorted({i['code'] for i in problems})}.")]
    if identity_hash(manifest) != bundle_sha256:
        issues.append(_issue("bundle_sha256_mismatch", SOURCE_MANIFEST_NAME,
                             "Manifest identity differs from the recorded bundle_sha256."))
    expected = {row["path"]: row for row in manifest["files"]}
    for name in sorted(set(members) - set(expected)):
        issues.append(_issue("extra_member", name, "Member is not in the manifest."))
    for name in sorted(set(expected) - set(members)):
        issues.append(_issue("missing_member", name, "Manifest file is not in the archive."))
    for name in sorted(set(expected) & set(members)):
        content, row = members[name], expected[name]
        if (len(content), hashlib.sha256(content).hexdigest()) != (row["size_bytes"],
                                                                    row["sha256"]):
            issues.append(_issue("member_mismatch", name, "Bytes differ from the manifest."))
    return issues


def _report(scope, issues, excluded, manifest=None, archive=None):
    sealed = not issues and manifest is not None and archive is not None
    bundle_sha256 = identity_hash(manifest) if sealed else None
    return {
        "artifact_kind": "s2_source_bundle_report", "schema_version": SCHEMA_VERSION,
        "scope": scope if scope in SCOPES else None, "producer": _producer(),
        "inputs": {"manifest_sha256": bundle_sha256}, "units": {"size_bytes": "bytes"},
        "level": "L1", "status": "passed" if sealed else "blocked",
        "bundle_sha256": bundle_sha256,
        "archive": ({"name": source_bundle_archive_name(bundle_sha256),
                     "size_bytes": len(archive),
                     "sha256": hashlib.sha256(archive).hexdigest()} if sealed else None),
        "manifest_name": SOURCE_MANIFEST_NAME,
        "files": len(manifest["files"]) if sealed else 0,
        "excluded": excluded, "issues": issues, "network": "none",
    }


def _write_new(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_bundle(root, out_dir, *, scope):
    """Collect, seal and write the bundle; returns the ``s2_source_bundle_report`` (L1).

    Writes ``<out_dir>/physloop-source-<bundle_sha256>.tar`` and ``.manifest.json`` with
    exclusive creation only when nothing blocks. Never raises for bad input.
    """
    issues = []
    if scope not in SCOPES:
        issues.append(_issue("invalid_scope", "scope", f"scope is one of {list(SCOPES)}."))
    if type(root) is not str or not os.path.isabs(root) or not os.path.isdir(root):
        issues.append(_issue("invalid_root", "root", "root is an existing absolute directory."))
    if type(out_dir) is not str or not os.path.isabs(out_dir) or not os.path.isdir(out_dir):
        issues.append(_issue("invalid_out_dir", "out_dir",
                             "out_dir is an existing absolute directory."))
    if issues:
        return _report(scope, issues, [])
    real_root, real_out = Path(os.path.realpath(root)), Path(os.path.realpath(out_dir))
    if real_out == real_root or real_root in real_out.parents:
        return _report(scope, [_issue("out_dir_inside_root", "out_dir",
                                      "Write the bundle outside the source root.")], [])
    files, excluded, issues = collect_sources(str(real_root))
    if not files and not issues:
        issues.append(_issue("missing_input", "root", "Nothing to bundle."))
    if issues:
        return _report(scope, issues, excluded)
    manifest = build_manifest(files, scope=scope)
    problems = validate_source_manifest(manifest)
    if problems:
        return _report(scope, [_issue("manifest_invalid", "manifest", str(sorted(
            {i["code"] for i in problems})))], excluded)
    archive = build_archive(files, manifest)
    if len(archive) > MAX_ARCHIVE_BYTES:
        return _report(scope, [_issue("archive_too_large", "archive",
                                      "Archive exceeds the bundle bound.")], excluded)
    report = _report(scope, [], excluded, manifest, archive)
    name = report["archive"]["name"]
    try:
        _write_new(real_out / name, archive)
        _write_new(real_out / (name[: -len(".tar")] + ".manifest.json"),
                   canonical_json_bytes(manifest))
    except OSError as exc:
        return _report(scope, [_issue("output_exists" if isinstance(exc, FileExistsError)
                                      else "output_unwritable", "out_dir",
                                      f"Output not written ({type(exc).__name__}).")], excluded)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="seal a source bundle")
    build.add_argument("--root", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--scope", required=True, choices=SCOPES)
    verify = commands.add_parser("verify", help="verify an archive against a bundle identity")
    verify.add_argument("archive")
    verify.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_bundle(args.root, args.out_dir, scope=args.scope)
        ok = report["status"] == "passed"
    else:
        try:
            data, _ = _read_regular(args.archive, MAX_ARCHIVE_BYTES)
            issues = verify_archive(data, args.bundle_sha256)
        except (OSError, ValueError) as exc:
            issues = [_issue("unreadable_archive", "archive", type(exc).__name__)]
        ok = not issues
        report = {"artifact_kind": "s2_source_bundle_verification",
                  "schema_version": SCHEMA_VERSION, "status": "passed" if ok else "blocked",
                  "bundle_sha256": args.bundle_sha256, "issues": issues}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
