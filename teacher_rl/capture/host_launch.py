"""Host-side launcher patch: run AlpaGym's host CLI with the PhysLoop capture entrypoint (H4).

Tickets: S4.3 (qualified capture without forking upstream), E15 (the capture code is part of the
run identity), lane-r3 §3 step 6 and §6 H4. Agent-authored under D-018/D-024.

``main`` (run as ``uv run --project <alpagym> python -B -m teacher_rl.capture.host_launch ...``
with ``PYTHONPATH=<staged capture tree>``):
0. refuses unless the launch is hash-bound recorded authority (G1; completeness-audit Safety:
   the launcher accepted any free-text reference, then ran the host CLI whose Hugging Face
   preflight uses the inherited ``HF_TOKEN`` and whose Wizard downloads gated NuRec scenes).
   ``verify_launch_authority`` takes the ``teacher_rl.launch.build_launch`` spec
   (``--launch-spec``), the registration it was compiled from (``--registration``), the bytes
   of the decision log (``--decisions``) and ``--decision-id``, and requires: the spec is a
   ``passed`` ``s4_launch_spec`` with ``scope`` ``actual`` whose ``launch_hash`` re-derives from
   its own fields (``launch_body``, the exact body ``build_launch`` hashes); the flags and Hydra
   overrides this process received equal the spec's argv after the module token, and the
   working directory equals its ``cwd``; the registration hashes
   (``gate.s2_identity.identity_hash``) to the spec's ``input_hashes.registration`` and has
   ``scope`` ``actual``; ``gate.s2_authority.verify_authority`` authorizes the decision for
   ``s4_launch_plan sha256:<launch_hash>`` and, as ``also_required``,
   ``s4_registration sha256:<registration hash>``; and ``verify_records`` finds
   ``records: terms:nurec_scenes`` and ``records: acquire:nurec_scenes`` in that entry.
   Validated registration: this process runs in AlpaGym's environment where
   ``teacher_rl.registration`` (numpy via ``gate.superiority``) is absent, so it does not
   re-run ``validate_registration``; it checks the registration's kind, schema version and
   top-level fields (mirrored constants pinned by a test; review G1-09), and
   ``python -m teacher_rl watch --execute`` (PhysLoop environment) does re-run
   ``validate_registration``. A ``passed`` spec exists only when
   ``compile_overrides``/``validate_registration`` passed at compile time, and the decision binds
   the registration's hash, so a different or edited registration is refused. The four
   authority flags are not part of the hashed argv (they point at the authority);
1. refuses without an explicit ``--action-authority-ref`` (recorded execution authority), an
   explicit ``--worker-credentials`` mode (no default) and an absolute ``run_root`` Hydra
   override (the pinned default ``./tmp/alpagym-runs`` gives Cosmos workers a relative
   ``custom.resolved_config_path``, which the capture entrypoint refuses);
2. verifies the staged capture tree byte-for-byte against its manifest; any ``__pycache__``
   directory or unlisted file is refused (an unchecked hash-based ``.pyc`` would run instead of
   the verified source), so the host process itself must run with ``-B``;
3. validates the capture configuration and exports only ``PYTHONPATH`` (the tree),
   ``PHYSLOOP_CAPTURE_CONFIG``, ``PYTHONDONTWRITEBYTECODE=1`` and
   ``PHYSLOOP_WORKER_CREDENTIALS``; ``launch_processes`` copies the host environment into every
   Cosmos replica (cosmos_rl/launcher/utility.py:806-814). A pre-existing ``PYTHONPATH`` is
   refused so the worker import closure is exactly upstream plus the tree. The host CLI keeps
   the inherited ``HF_TOKEN`` for its own preflight; each Cosmos role applies the credential
   mode on entry (``entrypoint.apply_worker_credentials``) before importing upstream;
4. wraps ``alpagym_host.run_lifecycle._build_cosmos_launcher_command`` (looked up as a module
   global by ``_build_cosmos_command``) so the final token ``alpagym_runtime.cosmos.entrypoint``
   becomes ``teacher_rl.capture.entrypoint``; any other command shape aborts;
5. registers the Hydra schema and calls the upstream ``alpagym_host.cli.main`` with the given
   Hydra overrides (the same sequence as the upstream ``__main__`` block).

Verification ceiling:
- ``rewrite_launcher_command``, ``stage_capture_tree``, ``verify_capture_tree``,
  ``capture_environment``, ``check_run_root``, ``launch_body``, ``verify_launch_authority``:
  **L1** (known answers and negative controls; ``launch_body`` is cross-checked against
  ``build_launch``'s own ``launch_hash`` on both routes).
- ``install_launcher_patch`` and ``main``: **L2** — the patch is exercised on the pinned
  ``_build_cosmos_launcher_command`` AST-extracted from the vendored ``run_lifecycle.py`` and the
  CLI sequence on fakes AST-checked against the vendored ``cli.py``/``config.py``.
- That ``uv run`` preserves ``PYTHONPATH`` into the workers and that Cosmos runs the substituted
  module in every role: **L3** (``qualify_on_host``).

Never: provisions, downloads, accepts a licence, embeds or forwards a credential, edits upstream
files, authenticates who wrote the decision bytes, or starts anything without hash-bound
recorded authority. Stdlib only at import time; ``gate.s2_authority`` and ``gate.s2_identity``
(both stdlib-only and in ``CAPTURE_FILES``) are imported inside functions.
"""

import argparse
import functools
import hashlib
import importlib
import inspect
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

CAPTURE_MODULE = "teacher_rl.capture.entrypoint"
MODULE_NAME = "teacher_rl.capture.host_launch"
UPSTREAM_MODULE = "alpagym_runtime.cosmos.entrypoint"
LAUNCHER_FUNCTION = "_build_cosmos_launcher_command"
LAUNCHER_PARAMS = ("config", "project_root", "no_sync", "worker_count", "worker_index",
                   "controller_port", "controller_url")
SENTINEL = "_physloop_capture_launcher"
CAPTURE_FILES = (
    # Rule-11 hashing import closure (stdlib only), needed by sha256_local snapshots.
    "gate/__init__.py",
    # G1: the hash-bound launch authority check runs in this (AlpaGym) process.
    "gate/s2_authority.py",
    "gate/s2_identity.py",
    "gate/tensor_manifest.py",
    "teacher_rl/__init__.py",
    "teacher_rl/capture/__init__.py",
    "teacher_rl/capture/entrypoint.py",
    "teacher_rl/capture/host_launch.py",
    "teacher_rl/capture/sink.py",
)
ENV_CONFIG = "PHYSLOOP_CAPTURE_CONFIG"
ENV_WORKER_CREDENTIALS = "PHYSLOOP_WORKER_CREDENTIALS"
WORKER_CREDENTIAL_MODES = ("strip_offline", "inherit_disclosed")
PLACEHOLDERS = {"tbd", "todo", "later", "____", "placeholder", "xxx", "n/a", "?"}
_AUTHORITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:/#()-]{2,199}")
LAUNCH_SPEC_KIND = "s4_launch_spec"
LAUNCH_ARTIFACT_KIND = "s4_launch_plan"
REGISTRATION_ARTIFACT_KIND = "s4_registration"
# Mirrors of teacher_rl.registration's ARTIFACT_KIND, SCHEMA_VERSION and TOP_FIELDS (that module
# needs numpy, absent from this process's import closure); a test pins them equal.
REGISTRATION_KIND = "s4_run_registration"
REGISTRATION_SCHEMA_VERSION = 2
REGISTRATION_FIELDS = ("artifact_kind", "schema_version", "scope", "registration_id",
                       "engineering", "owner")
NUREC_RECORDS = ("acquire:nurec_scenes", "terms:nurec_scenes")
AUTHORITY_FLAGS = ("--launch-spec", "--registration", "--decisions", "--decision-id")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LAUNCH_FIELDS = ("route", "argv", "cwd", "env_set", "env_allowlist", "secret_env",
                  "files_to_write", "effective_config_hash", "resolved_config_sha256",
                  "alpasim_cache_dir", "nurec_authorizations", "toml", "launch_hash",
                  "input_hashes", "scope", "status", "artifact_kind", "schema_version")


class LaunchRefused(RuntimeError):
    """A precondition for launching with capture is not met; nothing was started."""


def rewrite_launcher_command(command):
    """Replace the final upstream entrypoint token; refuse any other command shape."""
    if type(command) is not list or not all(type(item) is str for item in command):
        raise LaunchRefused("launcher command must be a list of strings")
    if command.count(UPSTREAM_MODULE) != 1 or command[-1] != UPSTREAM_MODULE:
        raise LaunchRefused(f"expected exactly one trailing {UPSTREAM_MODULE!r} token")
    if CAPTURE_MODULE in command:
        raise LaunchRefused("capture module already present")
    return [*command[:-1], CAPTURE_MODULE]


def install_launcher_patch(run_lifecycle):
    """Wrap ``_build_cosmos_launcher_command`` in place (idempotent; signature-checked)."""
    original = getattr(run_lifecycle, LAUNCHER_FUNCTION, None)
    if original is None or not callable(original):
        raise LaunchRefused(f"{LAUNCHER_FUNCTION} is missing from run_lifecycle")
    if getattr(original, SENTINEL, False):
        return "already_installed"
    names = tuple(inspect.signature(original).parameters)
    if names != LAUNCHER_PARAMS:
        raise LaunchRefused(f"{LAUNCHER_FUNCTION} signature {names} differs from the pinned "
                            f"{LAUNCHER_PARAMS}")

    @functools.wraps(original)
    def patched(*args, **kwargs):
        return rewrite_launcher_command(original(*args, **kwargs))

    setattr(patched, SENTINEL, True)
    setattr(run_lifecycle, LAUNCHER_FUNCTION, patched)
    return "installed"


def _read_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise LaunchRefused(f"not a regular file: {path}")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _tree_hash(files):
    lines = "".join(f"{f['path']}\0{f['sha256']}\0{f['bytes']}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def stage_capture_tree(source_root, dest_root, *, scope):
    """Copy exactly ``CAPTURE_FILES`` into a new directory; return its manifest.

    ``dest_root`` must not exist. Sources are read without following symlinks; the manifest
    binds every file's sha256 and size and a tree hash (part of the E15 run identity).
    """
    if scope not in ("synthetic", "public_source_fixture", "actual"):
        raise LaunchRefused("scope must be synthetic, public_source_fixture or actual")
    source_root, dest_root = Path(source_root), Path(dest_root)
    if os.path.lexists(dest_root):
        raise LaunchRefused(f"capture tree destination already exists: {dest_root}")
    files = []
    contents = {}
    for relative in CAPTURE_FILES:
        try:
            data = _read_regular(source_root / relative)
        except OSError as exc:
            raise LaunchRefused(f"cannot read {relative}: {exc}") from exc
        contents[relative] = data
        files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                      "bytes": len(data)})
    os.mkdir(dest_root, 0o755)
    for relative, data in contents.items():
        target = dest_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"artifact_kind": "s4_capture_tree", "schema_version": 1, "scope": scope,
            "producer": {"module": "teacher_rl.capture.host_launch",
                         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "inputs": [], "units": {"bytes": "file size in bytes"}, "issues": [],
            "files": files, "tree_hash": _tree_hash(files)}


def verify_capture_tree(tree_root, manifest):
    """The staged tree holds exactly the manifest's files with matching bytes; no links."""
    tree_root = Path(tree_root)
    issues = []
    expected = {f["path"]: f for f in manifest.get("files", [])}
    if set(expected) != set(CAPTURE_FILES):
        issues.append("manifest does not list exactly the capture files")
    if manifest.get("tree_hash") != _tree_hash(manifest.get("files", [])):
        issues.append("manifest tree_hash does not match its file list")
    found = set()
    for dirpath, dirnames, filenames in os.walk(tree_root, followlinks=False):
        for name in dirnames + filenames:
            full = Path(dirpath) / name
            info = os.lstat(full)
            relative = PurePosixPath(full.relative_to(tree_root).as_posix())
            if stat.S_ISLNK(info.st_mode):
                issues.append(f"symlink in capture tree: {relative}")
            elif stat.S_ISREG(info.st_mode):
                found.add(str(relative))
            elif not stat.S_ISDIR(info.st_mode):
                issues.append(f"special file in capture tree: {relative}")
            elif name == "__pycache__":
                # Python loads cached bytecode before (or instead of) the verified source; the
                # host runs with -B and workers with PYTHONDONTWRITEBYTECODE=1, so none is made.
                issues.append(f"bytecode cache directory in capture tree: {relative}")
    for extra in sorted(found - set(expected)):
        issues.append(f"unexpected file in capture tree: {extra}")
    for relative, entry in sorted(expected.items()):
        try:
            data = _read_regular(tree_root / relative)
        except (OSError, LaunchRefused) as exc:
            issues.append(f"missing or unreadable {relative}: {exc}")
            continue
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
            issues.append(f"bytes changed: {relative}")
    return {"status": "blocked" if issues else "passed", "issues": issues,
            "tree_hash": manifest.get("tree_hash")}


def capture_environment(base_env, *, capture_tree, capture_config_path, worker_credentials):
    """The only environment changes made for capture (names and non-secret values only).

    ``PYTHONPATH`` may already equal the capture tree (the host process itself is launched with
    it); any other existing value is refused. ``worker_credentials`` is required (no default).
    """
    tree, config = Path(capture_tree), Path(capture_config_path)
    if not tree.is_absolute() or not config.is_absolute():
        raise LaunchRefused("capture tree and config paths must be absolute")
    if base_env.get("PYTHONPATH") not in (None, "", str(tree)):
        raise LaunchRefused("PYTHONPATH names something other than the capture tree; refuse "
                            "an unknown import closure")
    if worker_credentials not in WORKER_CREDENTIAL_MODES:
        raise LaunchRefused(f"worker credential mode must be one of {WORKER_CREDENTIAL_MODES}")
    return {"PYTHONPATH": str(tree), ENV_CONFIG: str(config), "PYTHONDONTWRITEBYTECODE": "1",
            ENV_WORKER_CREDENTIALS: worker_credentials}


def check_run_root(overrides):
    """Exactly one Hydra ``run_root`` override naming a normalized absolute path."""
    values = []
    for override in overrides:
        key, sep, value = override.partition("=")
        if sep and key.lstrip("+") == "run_root":
            values.append(value[1:-1] if len(value) >= 2 and value[0] == value[-1] == "'"
                          else value)
    if len(values) != 1:
        raise LaunchRefused("the Hydra overrides must set run_root exactly once (compile it from "
                            "the registration's engineering.run_root; the pinned default "
                            "./tmp/alpagym-runs is relative)")
    path = PurePosixPath(values[0])
    if not path.is_absolute() or str(path) != values[0] or ".." in path.parts:
        raise LaunchRefused(f"run_root must be a normalized absolute path, got {values[0]!r}")
    return values[0]


def _authority(value):
    if type(value) is not str or not _AUTHORITY.fullmatch(value) or (
            value.strip().lower() in PLACEHOLDERS):
        raise LaunchRefused("--action-authority-ref must name the recorded execution "
                            "authority (decision ID and approval reference)")
    return value


def launch_body(spec):
    """The exact mapping ``teacher_rl.launch.build_launch`` hashes into ``launch_hash`` (L1)."""
    toml = spec["toml"]
    return {"route": spec["route"], "argv": spec["argv"], "cwd": spec["cwd"],
            "env_set": spec["env_set"], "env_allowlist": spec["env_allowlist"],
            "secret_env": spec["secret_env"],
            "files_to_write": [{k: f[k] for k in ("path", "sha256", "bytes")}
                               for f in spec["files_to_write"]],
            "effective_config_hash": spec["effective_config_hash"],
            "resolved_config_sha256": spec["resolved_config_sha256"],
            "base_cosmos_toml_sha256": None if toml is None else toml["base_sha256"],
            "alpasim_cache_dir": spec["alpasim_cache_dir"],
            "nurec_authorizations": spec["nurec_authorizations"]}


def _issue(code, message):
    return {"code": code, "path": "launch_authority", "severity": "blocking",
            "evidence_refs": [], "message": message}


def _spec_issues(spec):
    """Shape, status, scope, self-consistent hash and file contents of a launch spec."""
    from gate.s2_identity import identity_hash

    if type(spec) is not dict or any(key not in spec for key in _LAUNCH_FIELDS) or (
            spec["artifact_kind"] != LAUNCH_SPEC_KIND or spec["schema_version"] != 1):
        return [_issue("launch_spec_invalid", f"need a {LAUNCH_SPEC_KIND} v1 with "
                                              f"{list(_LAUNCH_FIELDS)}")]
    if spec["status"] != "passed" or type(spec["launch_hash"]) is not str or not (
            _HEX64.fullmatch(spec["launch_hash"])):
        # Status first (G3R): a blocked build_launch spec has argv/cwd None by design.
        return [_issue("launch_spec_not_passed", "only a passed spec carries a hash")]
    if type(spec["argv"]) is not list or not all(type(item) is str for item in spec["argv"]) or (
            type(spec["cwd"]) is not str):  # review G1-06: never a TypeError later
        return [_issue("launch_spec_invalid", "argv is a list of strings and cwd a string")]
    issues = []
    if spec["scope"] != "actual":
        issues.append(_issue("launch_scope_not_actual",
                             f"an actual launch needs scope 'actual', not {spec['scope']!r}"))
    try:
        derived = identity_hash(launch_body(spec))
        files_ok = all(
            type(f.get("content")) is str
            and hashlib.sha256(f["content"].encode()).hexdigest() == f["sha256"]
            and len(f["content"].encode()) == f["bytes"] for f in spec["files_to_write"])
    except (KeyError, TypeError, AttributeError, ValueError):
        return issues + [_issue("launch_spec_invalid", "launch fields are not hashable JSON")]
    if derived != spec["launch_hash"]:
        issues.append(_issue("launch_hash_mismatch",
                             "launch_hash does not re-derive from the spec's own fields"))
    if not files_ok:
        issues.append(_issue("launch_files_inconsistent",
                             "a file's content does not match its recorded sha256/bytes"))
    return issues


def verify_launch_authority(spec, registration, decisions_md, *, decision_id, launch_args=None,
                            cwd=None):
    """Hash-bound recorded authority for one actual S4 launch (L1; never raises).

    ``launch_args`` (this process's flags and overrides without the authority flags) and
    ``cwd`` are compared with the spec when given. Returns an ``s4_launch_authority_check``
    report; ``authorized`` is true only without any issue.
    """
    from gate.s2_authority import verify_authority, verify_records
    from gate.s2_identity import identity_hash

    issues = _spec_issues(spec)
    launch_hash = spec.get("launch_hash") if type(spec) is dict else None
    try:
        registration_hash = identity_hash(registration)
    except (TypeError, ValueError):
        registration_hash = None
    if type(registration) is not dict or registration_hash is None:
        issues.append(_issue("registration_invalid", "the registration must be a JSON object"))
    elif (registration.get("artifact_kind") != REGISTRATION_KIND
          or type(registration.get("schema_version")) is not int
          or registration["schema_version"] != REGISTRATION_SCHEMA_VERSION
          or any(key not in registration for key in REGISTRATION_FIELDS)):
        # Review G1-09: the shape validate_registration requires first (mirrored constants,
        # pinned equal by tests/test_s4_capture.py); a hash bound by a decision does not
        # make an arbitrary object a registration.
        issues.append(_issue("registration_invalid",
                             f"need a {REGISTRATION_KIND} v{REGISTRATION_SCHEMA_VERSION} with "
                             f"{list(REGISTRATION_FIELDS)}"))
    else:
        if registration.get("scope") != "actual":
            issues.append(_issue("registration_scope_not_actual",
                                 "an actual launch needs an actual-scope registration"))
        bound = (spec.get("input_hashes") or {}).get("registration") if type(
            spec) is dict and type(spec.get("input_hashes")) is dict else None
        if bound != registration_hash:
            issues.append(_issue("registration_mismatch",
                                 "the registration is not the one this spec was compiled from"))
    if not issues and launch_args is not None:
        argv = spec["argv"]
        tail = argv[argv.index(MODULE_NAME) + 1:] if MODULE_NAME in argv else None
        if tail != list(launch_args):
            issues.append(_issue("launch_argv_mismatch",
                                 "this process's flags and overrides differ from the spec argv"))
    if not issues and cwd is not None and spec["cwd"] != str(cwd):
        issues.append(_issue("launch_cwd_mismatch", f"run from {spec['cwd']}, not {cwd}"))
    authority = records = None
    if not issues:
        authority = verify_authority(
            decisions_md, decision_id=decision_id, artifact_kind=LAUNCH_ARTIFACT_KIND,
            artifact_sha256=launch_hash,
            also_required=((REGISTRATION_ARTIFACT_KIND, registration_hash),))
        if not authority["authorized"]:
            codes = sorted({i["code"] for i in authority["issues"]})
            issues.append(_issue("launch_not_authorized",
                                 f"{decision_id!r} does not authorize {LAUNCH_ARTIFACT_KIND} "
                                 f"{launch_hash} with {REGISTRATION_ARTIFACT_KIND} "
                                 f"{registration_hash} ({', '.join(codes)})"))
        else:
            records = verify_records(decisions_md, decision_id=decision_id, refs=NUREC_RECORDS)
            if not records["satisfied"]:
                issues.append(_issue("nurec_records_missing",
                                     f"{decision_id!r} lacks records: {records['missing_refs']}"
                                     " (the Wizard downloads gated NuRec scenes)"))
    return {"artifact_kind": "s4_launch_authority_check", "schema_version": 1,
            "producer": {"module": MODULE_NAME, "function": "verify_launch_authority"},
            "authorized": not issues, "decision_id": decision_id, "launch_hash": launch_hash,
            "registration_sha256": registration_hash,
            "decisions_sha256": None if authority is None else authority["decisions_sha256"],
            "entry_sha256": None if authority is None else authority["entry_sha256"],
            "issues": issues, "authenticity": "supplied_not_authenticated"}


def _load_json(path):
    from gate.s2_identity import read_json_bytes

    try:
        return read_json_bytes(_read_regular(Path(path)))
    except (OSError, ValueError) as exc:
        raise LaunchRefused(f"cannot read {path}: {type(exc).__name__}") from None


def main(argv=None, *, environ=None, import_module=None, set_argv=None, load_manifest=None,
         getcwd=None):
    """Verify, patch and hand off to the upstream host CLI with the remaining Hydra overrides."""
    from teacher_rl.capture.entrypoint import load_capture_config  # stdlib-only module

    environ = os.environ if environ is None else environ
    import_module = import_module or importlib.import_module
    parser = argparse.ArgumentParser(prog="teacher_rl.capture.host_launch")
    parser.add_argument("--action-authority-ref", required=True)
    parser.add_argument("--capture-tree", required=True)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--capture-config", required=True)
    parser.add_argument("--worker-credentials", required=True, choices=WORKER_CREDENTIAL_MODES)
    for flag in AUTHORITY_FLAGS:
        parser.add_argument(flag, required=True)
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    _authority(args.action_authority_ref)
    overrides = args.hydra_overrides[1:] if args.hydra_overrides[:1] == ["--"] else (
        args.hydra_overrides)
    if not overrides:
        raise LaunchRefused("no Hydra overrides given; compile them from the registration")
    check_run_root(overrides)
    try:
        decisions_md = _read_regular(Path(args.decisions))
    except OSError as exc:
        raise LaunchRefused(f"cannot read the decision log: {type(exc).__name__}") from None
    launch_args = ["--action-authority-ref", args.action_authority_ref,
                   "--capture-tree", args.capture_tree,
                   "--capture-manifest", args.capture_manifest,
                   "--capture-config", args.capture_config,
                   "--worker-credentials", args.worker_credentials, "--", *overrides]
    check = verify_launch_authority(_load_json(args.launch_spec), _load_json(args.registration),
                                    decisions_md, decision_id=args.decision_id,
                                    launch_args=launch_args,
                                    cwd=(getcwd or os.getcwd)())
    if not check["authorized"]:
        raise LaunchRefused("launch authority refused: "
                            + ", ".join(i["code"] for i in check["issues"]))
    if load_manifest is None:
        import json

        def load_manifest(path):
            return json.loads(_read_regular(Path(path)))

    tree_report = verify_capture_tree(args.capture_tree, load_manifest(args.capture_manifest))
    if tree_report["status"] != "passed":
        raise LaunchRefused(f"capture tree verification failed: {tree_report['issues']}")
    load_capture_config(args.capture_config)
    updates = capture_environment(environ, capture_tree=args.capture_tree,
                                  capture_config_path=args.capture_config,
                                  worker_credentials=args.worker_credentials)
    run_lifecycle = import_module("alpagym_host.run_lifecycle")
    install_launcher_patch(run_lifecycle)
    environ.update(updates)
    import_module("alpagym_host.config").register_config_schema()
    cli = import_module("alpagym_host.cli")
    (set_argv or _set_sys_argv)([CAPTURE_MODULE, *overrides])
    return cli.main()


def _set_sys_argv(values):
    sys.argv = list(values)


if __name__ == "__main__":
    main()
