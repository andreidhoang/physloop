"""Policy B re-export control CLI: compile the plan (dry run) and compare two checkpoints.

Serves S2.4 [C] (G3·R1, G7) under D-018/D-024; closes the completeness-audit gap "S2.4
producers" together with ``gate.s2_reexport``. Agent-authored.

``compile`` validates an ``s2_reexport_inventory`` and an ``s2_reexport_action_record`` and
prints the sealed ``s2_reexport_plan`` (``gate.s2_reexport.build_reexport_plan``); it never
executes a stage. The single guest runner ``scripts/s2_guest_bootstrap.py`` executes the plan
through the ``s2_reexport`` kind registered in ``gate.s2_bootstrap.PLAN_KINDS``.

``compare`` is the plan's control stage ``X1``: it builds ``gate.tensor_manifest`` manifests of
the release snapshot and the re-export (regular files only, opened without following symlinks
and memory-mapped read-only; the shard set is the one the ``--index`` file names) and writes
``release.manifest.json``, ``reexport.manifest.json`` and ``control.json``
(``gate.s2_reexport.reexport_control``) into a new ``--output-dir`` outside both roots. The two
roots must be disjoint: a checkpoint compared with itself would pass vacuously. A release
snapshot made of symlinks (an HF cache ``snapshots/`` directory) is refused by design; the
plan's teacher snapshot is a ``--local-dir`` download of regular files.

Path identity, not spelling. Every path is a portable absolute path
(``gate.s2_runtime.is_portable_absolute``: no ``//``, ``.``, ``..`` or empty component). Each
root is opened from ``/`` one component at a time without following symlinks, and disjointness
is decided on directory identity ``(st_dev, st_ino)`` over each directory's physical ancestor
chain (walked through ``..``), so a case variant on a case-insensitive filesystem or a bind
mount of the same directory is refused like the same string. The output directory's existing
ancestors are opened the same way (a symlink or file component is ``invalid_output_path``),
refused when that chain contains either root, and its missing components are created relative
to the no-follow parent descriptor; the three files are created ``O_EXCL | O_NOFOLLOW`` inside
it. A bind mount cannot be exercised on CPU here; it is covered by the identity rule (L2).

Exit codes: 0 compiled / control ``passed``; 1 refused plan / control ``blocked``; 2 refused
input (unreadable or malformed file, symlink, bad argument, overlapping roots, existing,
misplaced or symlinked output, or any ``OSError`` while walking a root's or the output's
ancestry, such as a directory without search permission) with a JSON error on stderr.

Verification level: L1 on CPU temporary directories. On an actual release and re-export the
control is L3 and qualifies only those bytes. This script never imports torch or model code,
never downloads, never writes into a checkpoint, never runs a subprocess and never contacts a
network. Usage::

    python scripts/s2_reexport.py compile --inventory INV.json --action-record REC.json \
        --phase verify --phase convert --phase control [--output PLAN.json]
    python scripts/s2_reexport.py compare --release-root /data/teacher \
        --reexport-root /data/teacher-reexport --index model.safetensors.index.json \
        --scope actual --output-dir /data/evidence/s2-reexport-control
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import canonical_json_bytes, read_json_bytes
from gate.s2_reexport import CONTROL_FILES, PHASES, SCOPES, build_reexport_plan, reexport_control
from gate.s2_runtime import is_portable_absolute, paths_overlap
from gate.tensor_manifest import build_manifest_from_directory


class CliError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    compile_ = sub.add_parser("compile", help="compile the re-export plan (never executes)")
    compile_.add_argument("--inventory", required=True, type=Path)
    compile_.add_argument("--action-record", required=True, type=Path)
    compile_.add_argument("--phase", action="append", required=True, dest="phases",
                          choices=PHASES)
    compile_.add_argument("--output", type=Path, help="new file for the plan (else stdout)")
    compare = sub.add_parser("compare", help="manifest both checkpoints and run the control")
    compare.add_argument("--release-root", required=True, type=Path)
    compare.add_argument("--reexport-root", required=True, type=Path)
    compare.add_argument("--index", required=True, help="shard index file name at each root")
    compare.add_argument("--scope", required=True, choices=SCOPES)
    compare.add_argument("--output-dir", required=True, type=Path)
    return parser


def _read_json(path):
    try:
        return read_json_bytes(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise CliError("unreadable_input", f"{path}: {exc}") from None


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _absolute(path, what):
    text = str(path)
    if not is_portable_absolute(text):
        raise CliError("invalid_path", f"{what} must be a portable absolute path (no '//', "
                       "'.', '..' or empty component)")
    return Path(text)


def _walk(path):
    """Open ``path`` from ``/`` without following symlinks.

    Returns ``(fd, missing)``: a descriptor of the deepest existing directory and the
    components below it that do not exist. Raises ``OSError`` on a symlink or non-directory.
    """
    parts = Path(path).parts[1:]
    fd = os.open("/", _DIR_FLAGS)
    try:
        for index, part in enumerate(parts):
            try:
                following = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                return fd, list(parts[index:])
            os.close(fd)
            fd = following
        return fd, []
    except BaseException:
        os.close(fd)
        raise


def _identity(fd):
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def _ancestry(fd):
    """``(st_dev, st_ino)`` of the directory ``fd`` and of every physical ancestor."""
    chain = [_identity(fd)]
    current = os.dup(fd)
    try:
        while True:
            parent = os.open("..", _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = parent
            identity = _identity(current)
            if identity == chain[-1]:
                return chain
            chain.append(identity)
    finally:
        os.close(current)


def _root(path):
    try:
        fd, missing = _walk(path)
    except OSError as exc:
        raise CliError("unreadable_checkpoint", f"{path}: {exc}") from None
    try:
        if missing:
            raise CliError("unreadable_checkpoint", f"{path}: no such directory")
        return _identity(fd), _ancestry(fd)
    except OSError as exc:  # e.g. a root without search permission: openat(fd, '..')
        raise CliError("unreadable_checkpoint", f"{path}: {exc}") from None
    finally:
        os.close(fd)


def _output_parent(output, roots):
    """No-follow fd of the output's deepest existing ancestor and the missing components."""
    try:
        fd, missing = _walk(output)
    except OSError as exc:
        raise CliError("invalid_output_path", f"{output}: --output-dir must be reached through "
                       f"real directories, no symlink or file component ({exc})") from None
    try:
        if not missing:
            raise CliError("output_exists", f"{output} exists; outputs are never overwritten")
        chain = _ancestry(fd)
    except OSError as exc:
        os.close(fd)
        raise CliError("invalid_output_path", f"{output}: the output's ancestry cannot be "
                       f"walked through '..' ({exc})") from None
    except BaseException:
        os.close(fd)
        raise
    if any(root in chain for root in roots):
        os.close(fd)
        raise CliError("output_inside_checkpoint", "--output-dir must lie outside both "
                       "checkpoint roots")
    return fd, missing


def _make_output(output, roots):
    """Create the output directory under no-follow parent fds; return its fd."""
    fd, missing = _output_parent(output, roots)
    try:
        for part in missing:
            os.mkdir(part, dir_fd=fd)
            following = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = following
        return fd
    except OSError as exc:
        os.close(fd)
        raise CliError("output_not_writable", f"{output}: {exc}") from None
    except BaseException:
        os.close(fd)
        raise


def _write_new_at(dir_fd, name, payload):
    try:
        stream = os.fdopen(os.open(name, _FILE_FLAGS, 0o666, dir_fd=dir_fd), "wb")
    except OSError as exc:
        raise CliError("output_not_writable", f"{name}: {exc}") from None
    with stream:
        stream.write(canonical_json_bytes(payload) + b"\n")


def _write_new(path, payload):
    try:
        with open(path, "xb") as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
    except OSError as exc:
        raise CliError("output_not_writable", f"{path}: {exc}") from None


def _compile(args):
    inventory, record = _read_json(args.inventory), _read_json(args.action_record)
    plan = build_reexport_plan(inventory, record, phases=args.phases)
    if args.output is None:
        sys.stdout.write(canonical_json_bytes(plan).decode() + "\n")
    else:
        _write_new(_absolute(args.output, "--output"), plan)
    return 0 if plan["compiled"] else 1


def _compare(args):
    release = _absolute(args.release_root, "--release-root")
    reexport = _absolute(args.reexport_root, "--reexport-root")
    output = _absolute(args.output_dir, "--output-dir")
    if paths_overlap(release, reexport):
        raise CliError("roots_overlap", "the release and re-export roots must be disjoint")
    for root in (release, reexport):
        if paths_overlap(output, root):
            raise CliError("output_inside_checkpoint", "--output-dir must lie outside both "
                           "checkpoint roots")
    if os.path.lexists(output):
        raise CliError("output_exists", f"{output} exists; outputs are never overwritten")
    # Identity, not spelling: case variants, bind mounts and aliases name one directory.
    (release_id, release_chain), (reexport_id, reexport_chain) = _root(release), _root(reexport)
    if release_id in reexport_chain or reexport_id in release_chain:
        raise CliError("roots_overlap", "the release and re-export roots must be disjoint "
                       "directories (same or nested directory under another name)")
    os.close(_output_parent(output, (release_id, reexport_id))[0])
    manifests = []
    for root in (release, reexport):
        try:
            manifests.append(build_manifest_from_directory(
                str(root), None, scope=args.scope, index_name=args.index))
        except ValueError as exc:
            raise CliError("unreadable_checkpoint", f"{root}: {exc}") from None
    control = reexport_control(*manifests)
    out_fd = _make_output(output, (release_id, reexport_id))
    try:
        for name, payload in zip(CONTROL_FILES, (*manifests, control), strict=True):
            _write_new_at(out_fd, name, payload)
    finally:
        os.close(out_fd)
    differences = sum(len(v) for k, v in control["differences"].items() if k != "changed")
    print(f"{control['status']}: {control['tensors']['identical_count']} identical tensors, "
          f"{differences} differences; files_identical={control['files']['files_identical']}")
    return 0 if control["status"] == "passed" else 1


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return _compile(args) if args.command == "compile" else _compare(args)
    except CliError as exc:
        sys.stderr.write(json.dumps({"error": exc.code, "message": str(exc)}) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
