"""Read-only parity-ladder CLI over two S4 checkpoint directories (A10; E02, E04, E16).

Serves S4 engineering preparation (S4.1-S4.4, S4.6 [C]; G9, G7) under D-024. Agent-authored.

Builds rule-11 manifests of both checkpoints with ``gate.tensor_manifest`` (no-follow,
memory-mapped, torch-free) and runs ``teacher_rl.parity.ladder_from_manifests``: structure,
then bitwise identity, then - only for differing tensors and only when torch and safetensors
are importable - per-element error and ULP distance on tensors loaded one at a time with
``safetensors.safe_open`` and re-hashed against the manifest before use. The tolerance is a
registered owner input read from ``--tolerance`` (JSON with ``registration_ref``, ``atol``,
``rtol``, ``max_ulp``); without it differences are ``tolerance_unregistered``. Action-level
parity is not computed here (``qualify_on_host``). ``--cross-dtype
reference_cast_to_candidate`` declares a float cast between the checkpoints (the reference is
rounded to the candidate's stored dtype and measured there, see ``teacher_rl.parity``);
``--name`` (repeatable) restricts the ladder to declared tensors, e.g. E02's
``requires_parity_ladder`` when comparing a release with its inference conversion.

Exit codes: 0 report ``passed`` with verdict ``identical`` or
``within_registered_tolerance``; 1 report ``blocked`` (e.g. structure mismatch, bytes changed
since the manifest) or ``qualify_on_host`` (elementwise rung deferred: torch or safetensors
not importable); 2 refused with a JSON error on stderr (bad tolerance or argument,
unreadable checkpoint, existing output, output inside either checkpoint root, output not
writable, report not strict JSON); 3 report ``passed`` whose verdict is ``exceeds`` or
``tolerance_unregistered``.

Verification level: L1 on CPU temporary directories; on actual checkpoints L3. This script
never imports torch at module level, never loads model code, never downloads, never writes
into a checkpoint, never runs a subprocess or network call and never chooses a tolerance.
Usage::

    python scripts/s4_parity_check.py --scope actual \
        --reference-root REF --reference-index model.safetensors.index.json \
        --candidate-root CAND --candidate-index model.safetensors.index.json \
        --tolerance registered_tolerance.json --output /tmp/parity.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate.s2_identity import read_json_bytes
from gate.tensor_manifest import build_manifest_from_directory
from teacher_rl.parity import (
    CROSS_DTYPE_MODES,
    SCOPES,
    ladder_from_manifests,
    validate_tolerance,
)

PASSING_VERDICTS = ("identical", "within_registered_tolerance")


class CliError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for side in ("reference", "candidate"):
        parser.add_argument(f"--{side}-root", required=True, type=Path)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{side}-file", action="append", dest=f"{side}_files")
        group.add_argument(f"--{side}-index", dest=f"{side}_index")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--tolerance", type=Path)
    parser.add_argument("--cross-dtype", choices=CROSS_DTYPE_MODES)
    parser.add_argument("--name", action="append", dest="names",
                        help="restrict the ladder to this tensor name (repeatable)")
    parser.add_argument("--output", type=Path)
    return parser


def manifest(root, *, files, index, scope):
    """Rule-11 manifest of one checkpoint directory (no-follow, read-only)."""
    try:
        return build_manifest_from_directory(root, files, scope=scope, index_name=index)
    except ValueError as exc:
        raise CliError("manifest_read_failed", f"{root}: {exc}") from exc


def loader(reference_root, candidate_root, reference_manifest, candidate_manifest):
    """Return ``load(side, name)`` reading one tensor at a time with safetensors."""
    from safetensors import safe_open

    roots = {"reference": Path(reference_root), "candidate": Path(candidate_root)}
    manifests = {"reference": reference_manifest, "candidate": candidate_manifest}

    def load(side, name):
        path = roots[side] / manifests[side]["tensors"][name]["file"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    return load


def _torch_available():
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _read_tolerance(path):
    if path is None:
        return None
    try:
        tolerance = read_json_bytes(Path(path).read_bytes())
        validate_tolerance(tolerance)
    except (OSError, ValueError) as exc:
        raise CliError("tolerance_invalid", f"{path}: {exc}") from exc
    return tolerance


def _check_output(output, roots):
    """Refuse an existing output and any output that resolves inside a checkpoint root."""
    if output is None:
        return
    try:
        if output.exists() or output.is_symlink():
            raise CliError("output_exists", f"{output} exists; never overwritten")
        target = output.absolute().resolve()
        bases = [Path(root).absolute().resolve() for root in roots]
    except (OSError, RuntimeError) as exc:
        raise CliError("output_invalid", f"cannot resolve {output}: {exc}") from exc
    for base in bases:
        if target == base or base in target.parents:
            raise CliError("output_inside_checkpoint", "never write into a checkpoint root")


def _emit(report, output):
    try:
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def _exit_code(report):
    if report["status"] != "passed":
        return 1
    return 0 if report["verdict"] in PASSING_VERDICTS else 3


def main(argv=None):
    args = _parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        _check_output(args.output, (args.reference_root, args.candidate_root))
        tolerance = _read_tolerance(args.tolerance)
        ref = manifest(args.reference_root, files=args.reference_files,
                       index=args.reference_index, scope=args.scope)
        cand = manifest(args.candidate_root, files=args.candidate_files,
                        index=args.candidate_index, scope=args.scope)
        load = None
        if _torch_available() and ref["status"] == cand["status"] == "passed":
            load = loader(args.reference_root, args.candidate_root, ref, cand)
        try:
            report = ladder_from_manifests(ref, cand, tolerance=tolerance, load=load,
                                           cross_dtype=args.cross_dtype, names=args.names)
        except ValueError as exc:
            raise CliError("argument_invalid", str(exc)) from exc
        except OSError as exc:
            raise CliError("checkpoint_read_failed", str(exc)) from exc
        _emit(report, args.output)
    except CliError as exc:
        sys.stderr.write(json.dumps({
            "artifact_kind": "s4_cli_error", "schema_version": 1, "status": "blocked",
            "issues": [{"code": exc.code, "path": "argv", "severity": "error",
                        "evidence_refs": [], "message": str(exc)}],
        }, sort_keys=True) + "\n")
        return 2
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
