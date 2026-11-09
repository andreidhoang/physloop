"""Read-only tensor manifest and diff CLI over S4 checkpoint directories (A10; E02, E03).

Serves S4 engineering preparation (S4.1-S4.4, S4.6 [C]; G9, G7) under D-024. Agent-authored.

``manifest`` hashes every tensor of a safetensors checkpoint (single files or an indexed
shard set) with ``gate.tensor_manifest`` under build-spec section 2 rule 11: files are opened
without following symlinks and memory-mapped read-only, so a 10B checkpoint is never loaded
into RAM. ``diff`` compares two manifests or ``teacher_rl.params`` snapshots with
``teacher_rl.params.diff`` (changed tensors, changed fractions by prefix). ``links``
records an AlpaGym-converted bundle whose weight shards are symlinks to the release shards
(lane R3 section 2), which the no-follow ``manifest`` reader refuses by design: it hashes
the release with the same no-follow reader, then ``lstat``/``readlink``s (never follows)
each release file name inside the bundle and records whether it is a symlink whose
target is canonical (``teacher_rl.export_audit.link_target_problem``: no ``.``, empty or
inner ``..`` component), lexically normalizes to exactly ``<release root>/<name>`` and
physically resolves (``realpath`` plus ``st_dev``/``st_ino``, recorded as
``resolved_target``) to that release file, or a regular file whose bytes hash to the
release file (a copy). Both roots must be absolute-path symlink-free.
``teacher_rl.export_audit.check_zero_update_export`` consumes the record as ``gym_bundle``
and re-binds it to the release manifest it is given. Output goes to stdout or to a new
``--output`` file outside every checkpoint root (never overwritten).

Exit codes: 0 artifact with status ``passed``; 1 artifact produced but ``blocked``; 2 refused
(unreadable input, symlink where a regular file is required, bad argument, existing output,
output inside a checkpoint, output not writable) with a JSON error on stderr.

Verification level: L1 on CPU temporary directories; on an actual checkpoint (release,
converted bundle, Cosmos export, inference conversion) the result is L3 and qualifies only
that checkpoint's bytes. This script never imports torch or model code, never downloads,
never writes into a checkpoint, never reads bytes through a symlink (``links`` only
resolves and stats link targets) and never runs a
subprocess or network call. Usage::

    python scripts/s4_snapshot_params.py manifest --root CKPT --scope actual \
        --index model.safetensors.index.json --output /tmp/release.manifest.json
    python scripts/s4_snapshot_params.py diff before.json after.json
    python scripts/s4_snapshot_params.py links --bundle-root BUNDLE --release-root RELEASE \
        --scope actual --index model.safetensors.index.json --output /tmp/bundle.links.json
"""

import argparse
import hashlib
import json
import os
import posixpath
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import identity_hash, read_json_bytes
from gate.tensor_manifest import build_manifest_from_directory
from teacher_rl.export_audit import link_target_problem
from teacher_rl.params import DEFAULT_PREFIXES, SCOPES, diff

LINKS_KIND = "s4_bundle_links"
LINKS_SCHEMA_VERSION = 1
_CHUNK = 1 << 20


class CliError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="hash every tensor of a checkpoint directory")
    manifest.add_argument("--root", required=True, type=Path)
    manifest.add_argument("--scope", required=True, choices=SCOPES)
    group = manifest.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", action="append", dest="files",
                       help="safetensors file relative to --root (repeatable)")
    group.add_argument("--index", help="shard index file name at --root")
    manifest.add_argument("--output", type=Path)
    compare = sub.add_parser("diff", help="diff two manifests or snapshots")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--prefix", action="append", dest="prefixes")
    compare.add_argument("--output", type=Path)
    links = sub.add_parser("links", help="record a symlinked bundle against its release")
    links.add_argument("--bundle-root", required=True, type=Path)
    links.add_argument("--release-root", required=True, type=Path)
    links.add_argument("--scope", required=True, choices=SCOPES)
    group = links.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", action="append", dest="files",
                       help="release safetensors file relative to --release-root (repeatable)")
    group.add_argument("--index", help="release shard index file name at --release-root")
    links.add_argument("--output", type=Path)
    return parser


def _read_artifact(path):
    try:
        return read_json_bytes(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise CliError("input_invalid", f"cannot read {path}: {exc}") from exc


def _symlink_free(root, label):
    """Return the absolute root as a string, refusing traversal and symlinked components."""
    absolute = Path(root).absolute()
    if ".." in absolute.parts:
        raise CliError("root_invalid", f"{label} root cannot contain parent traversal")
    try:
        real = os.path.realpath(absolute, strict=True)
    except OSError as exc:
        raise CliError("root_invalid", f"cannot open {label} root: {exc}") from exc
    if real != str(absolute) or not os.path.isdir(real):
        raise CliError("root_invalid", f"{label} root must be a directory reached without links")
    return str(absolute)


def _sha256_no_follow(directory_fd, name):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _resolve_link(directory_fd, bundle_root, name, expected):
    """Return (physical path of the bundle entry or ``None``, same inode as ``expected``)."""
    try:
        resolved = os.path.realpath(posixpath.join(bundle_root, name), strict=True)
        linked = os.stat(name, dir_fd=directory_fd)  # follows the link chain: metadata only
        release = os.lstat(expected)
    except OSError:
        return None, False
    return resolved, stat.S_ISREG(release.st_mode) and os.path.samestat(linked, release)


def _link_entry(directory_fd, bundle_root, name, expected):
    """Return the entry of ``name`` and whether a symlink is physically ``expected``."""
    none = {"target": None, "normalized_target": None, "resolved_target": None}
    try:
        info = os.lstat(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return {"kind": "missing", **none, "sha256": None}, False
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(name, dir_fd=directory_fd)
        joined = target if posixpath.isabs(target) else posixpath.join(bundle_root, target)
        resolved, same_file = _resolve_link(directory_fd, bundle_root, name, expected)
        return {"kind": "symlink", "target": target,
                "normalized_target": posixpath.normpath(joined),
                "resolved_target": resolved, "sha256": None}, same_file
    if stat.S_ISREG(info.st_mode):
        return {"kind": "regular_file", **none,
                "sha256": _sha256_no_follow(directory_fd, name)}, False
    return {"kind": "other", **none, "sha256": None}, False


def _link_issue(code, name, message):
    return {"code": code, "path": f"entries/{name}", "severity": "error", "evidence_refs": [],
            "message": message}


def bundle_links(bundle_root, release_root, *, files, index, scope):
    """Return the ``s4_bundle_links`` record of a converted bundle (L1 on CPU temp dirs)."""
    release_abs = _symlink_free(release_root, "release")
    bundle_abs = _symlink_free(bundle_root, "bundle")
    try:
        release = build_manifest_from_directory(release_abs, files, scope=scope,
                                                index_name=index)
    except ValueError as exc:
        raise CliError("manifest_read_failed", str(exc)) from exc
    if release["status"] != "passed":
        raise CliError("release_manifest_blocked", "the release manifest has blocking issues")
    release_files = dict(sorted(release["input_hashes"]["files"].items()))
    if any("/" in name for name in release_files):
        raise CliError("argument_invalid", "links records bare shard names at the bundle root")
    entries, issues = {}, []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        directory_fd = os.open(bundle_abs, flags)
    except OSError as exc:
        raise CliError("root_invalid", f"cannot open bundle root: {exc}") from exc
    try:
        for name, sha in release_files.items():
            expected = posixpath.join(release_abs, name)
            try:
                entry, same_file = _link_entry(directory_fd, bundle_abs, name, expected)
            except OSError as exc:
                raise CliError("bundle_read_failed", f"{name}: {exc}") from exc
            entries[name] = entry
            if entry["kind"] == "missing":
                issues.append(_link_issue("bundle_file_missing", name, "absent from the bundle"))
            elif entry["kind"] == "symlink" and link_target_problem(entry["target"]):
                issues.append(_link_issue("bundle_link_target_invalid", name,
                                          link_target_problem(entry["target"])))
            elif entry["kind"] == "symlink" and entry["normalized_target"] != expected:
                issues.append(_link_issue("bundle_link_target_mismatch", name,
                                          f"links to {entry['normalized_target']!r}, "
                                          f"expected {expected!r}"))
            elif entry["kind"] == "symlink" and (entry["resolved_target"] != expected
                                                 or not same_file):
                issues.append(_link_issue("bundle_link_physical_mismatch", name,
                                          f"resolves to {entry['resolved_target']!r}, not "
                                          f"the release file {expected!r}"))
            elif entry["kind"] == "regular_file" and entry["sha256"] != sha:
                issues.append(_link_issue("bundle_file_bytes_differ", name,
                                          "regular file whose bytes differ from the release"))
            elif entry["kind"] == "other":
                issues.append(_link_issue("bundle_entry_not_file_or_link", name,
                                          "neither a symlink nor a regular file"))
    finally:
        os.close(directory_fd)
    return {
        "artifact_kind": LINKS_KIND,
        "schema_version": LINKS_SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "scripts.s4_snapshot_params",
                     "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "input_hashes": {"release_manifest": identity_hash(release)},
        "units": {},
        "status": "blocked" if issues else "passed",
        "bundle_root": bundle_abs,
        "release_root": release_abs,
        "release_tensor_set_sha256": release["tensor_set_sha256"],
        "release_files": release_files,
        "entries": entries,
        "issues": issues,
    }


def build(argv):
    """Return the artifact for ``argv`` or raise ``CliError`` (no output written)."""
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        try:
            return build_manifest_from_directory(
                args.root, args.files, scope=args.scope, index_name=args.index
            )
        except ValueError as exc:
            raise CliError("manifest_read_failed", str(exc)) from exc
    if args.command == "links":
        return bundle_links(args.bundle_root, args.release_root, files=args.files,
                            index=args.index, scope=args.scope)
    prefixes = tuple(args.prefixes) if args.prefixes else DEFAULT_PREFIXES
    try:
        return diff(_read_artifact(args.before), _read_artifact(args.after), prefixes=prefixes)
    except ValueError as exc:
        if isinstance(exc, CliError):
            raise
        raise CliError("argument_invalid", str(exc)) from exc


def _roots(args):
    if args.command == "manifest":
        return (args.root,)
    if args.command == "links":
        return (args.bundle_root, args.release_root)
    return ()


def _check_output(output, roots):
    if output is None:
        return
    try:
        if output.exists() or output.is_symlink():
            raise CliError("output_exists", f"{output} exists; outputs are never overwritten")
        target = output.absolute().resolve()
        bases = [Path(root).absolute().resolve() for root in roots]
    except (OSError, RuntimeError) as exc:
        raise CliError("output_invalid", f"cannot resolve {output}: {exc}") from exc
    for base in bases:
        if target == base or base in target.parents:
            raise CliError("output_inside_checkpoint", "never write into a checkpoint root")


def _emit(artifact, output):
    try:
        text = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as exc:
        raise CliError("report_not_serializable", str(exc)) from exc
    if output is None:
        sys.stdout.write(text)
        return
    try:
        with open(output, "x", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        raise CliError("output_write_failed", str(exc)) from exc


def _error(exc):
    payload = {
        "artifact_kind": "s4_cli_error",
        "schema_version": 1,
        "status": "blocked",
        "issues": [{"code": exc.code, "path": "argv", "severity": "error", "evidence_refs": [],
                    "message": str(exc)}],
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    return 2


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(argv)
    try:
        _check_output(args.output, _roots(args))
        artifact = build(argv)
        _emit(artifact, args.output)
    except CliError as exc:
        return _error(exc)
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
