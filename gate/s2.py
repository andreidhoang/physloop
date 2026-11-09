"""Offline S2 tooling. Does not provision, download assets, or launch experiments.

Every subcommand reads local JSON (strict: no duplicate keys or non-finite numbers),
calls one pure or dry-run function and prints one JSON document. Exit code 0 means
the command's report was accepted, 2 that it was rejected or the input was invalid;
input, file and structural errors (including attribute, key and index errors from a
malformed input) print ``{"error": ..., "spend_authorized": false}``.

Acceptance per command (``accepted``):

==========================  ==================================================
command                     accepted when
==========================  ==================================================
preflight                   ``ready_for_planning`` (never launch authority)
normalize                   ``admitted`` (whole batch)
matrix, schedule,           always (pure arithmetic / schedule construction)
cost-table
ingest                      ``admitted``
inspect-capture             ``capture_inspection_passed``
prepare-launch,             ``prepared`` (commands are printed, never executed)
prepare-pilot,
prepare-full-sweep
compile-bootstrap           ``compiled`` (plan only; nothing runs)
seal                        ``sealed`` (writes only the manifest and seal marker
                            inside ``--root``)
verify-bundle               ``verified``
every other command         ``status == "passed"``
==========================  ==================================================

``describe-repeats`` stays ``blocked`` (exit 2) until a qualified reliability method
is registered: its descriptive statistics are still printed. Request files for the
newer commands are JSON objects whose keys are the function's arguments; unknown or
missing required keys are input errors. No subcommand rents, downloads, accepts
terms, launches a workload or fills an owner value.
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA, normalize_summary
from gate.s2_capture import MAX_ASL_BYTES
from gate.s2_identity import identity_hash, read_artifact, read_json_bytes
from gate.s2_preflight import validate_s2_preflight
from gate.s2_schedule import active_matrix_report, build_schedule

MAX_ASL_BYTES_HELP = (
    "streamed, hash-verified; agent technical default (D-018), provisional until the first "
    "measured ASL"
)


def cost_table():
    """Arithmetic scenarios from the dated plan and public-price audit, not forecasts."""
    return {
        "price_check_date": "2026-10-31",
        "basis": "Existing 136–340 GPU-hour four-sweep estimate; GPU-only; reprice before rental",
        "spend_authorized": False,
        "rows": [
            {
                "gpu_hours": hours,
                "rate_per_gpu_hour_usd": float(rate),
                "compute_cost_usd": float(hours * Decimal(rate)),
            }
            for hours in (136, 340)
            for rate in ("1.80", "2.49")
        ],
    }


def _read(path):
    return read_json_bytes(Path(path).read_bytes())


def _arguments(request, required, optional=()):
    """Exact request keys: every required key present, nothing outside required+optional."""
    if type(request) is not dict:
        raise ValueError("Request must be a JSON object")
    missing = set(required) - set(request)
    unexpected = set(request) - set(required) - set(optional)
    if missing or unexpected:
        raise ValueError(f"Request keys: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    return dict(request)


def _status(result):
    return result.get("status") == "passed"


def _describe_validity(args):
    from gate.s2_diagnostics import describe_validity

    request = _arguments(_read(args.request), ("ledger_report", "observations"),
                         ("records", "validity_definition", "scope"))
    return describe_validity(request.pop("ledger_report"), request.pop("observations"),
                             **request)


def _audit_population(args):
    from gate.s2_population import audit_population

    request = _arguments(_read(args.request), ("manifest",), ("roles", "overlap_rule"))
    return audit_population(request.pop("manifest"), **request)


def _describe_repeats(args):
    from gate.s2_probes import describe_repeats

    request = _arguments(_read(args.request), ("observations", "scope"),
                         ("method", "paired_policies"))
    if type(request.get("paired_policies")) is list:
        request["paired_policies"] = tuple(request["paired_policies"])
    return describe_repeats(request.pop("observations"), **request)


def _probe_side(value, root):
    side = _arguments(value, ("receipt", "payload"))
    return {"receipt": side["receipt"], "payload": read_artifact(root, side["payload"])}


def _compare_probes(args):
    from gate.s2_probes import compare_probe_artifacts

    request = _arguments(_read(args.request), ("a", "b", "kind"), ("tolerance",))
    return compare_probe_artifacts(_probe_side(request["a"], args.root),
                                   _probe_side(request["b"], args.root), request["kind"],
                                   tolerance=request.get("tolerance"))


def _summarize_telemetry(args):
    from gate.s2_telemetry import summarize_telemetry

    request = _arguments(_read(args.request), (
        "samples", "allocation", "rollouts", "clock_metadata", "max_sample_gap_us", "scope"),
        ("stabilization_rule",))
    return summarize_telemetry(request.pop("samples"), request.pop("allocation"),
                               request.pop("rollouts"), **request)


def _run_sensitivity(args):
    from gate.s2_power import run_sensitivity

    request = _arguments(_read(args.request), ("design", "scenarios", "n_sim", "seed"))
    return run_sensitivity(**request)


def _evaluate_registered(args):
    from gate.s2_gate import evaluate_registered

    request = _arguments(_read(args.request),
                         ("records", "registration", "scope", "adapter_metric"))
    return evaluate_registered(request.pop("records"), request.pop("registration"), **request)


def _inspect_actual(args):
    from gate.s2_actual import inspect_actual_unit

    request = _arguments(_read(args.request), ("receipts", "binding", "scope"))
    return inspect_actual_unit(args.root, request["receipts"], request["binding"],
                               scope=request["scope"])


def _validate_inventory(args):
    from gate.s2_runtime import validate_runtime_inventory

    return validate_runtime_inventory(_read(args.inventory))


def _compile_bootstrap(args):
    from gate.s2_bootstrap import build_execution_plan

    return build_execution_plan(_read(args.inventory), _read(args.action_record),
                                phases=list(args.phases))


def _build_tensor_manifest(args):
    from gate.tensor_manifest import build_manifest_from_directory

    return build_manifest_from_directory(args.root, args.files or None, scope=args.scope,
                                         index_name=args.index)


def _compare_tensor_manifests(args):
    from gate.tensor_manifest import compare_manifests

    return compare_manifests(_read(args.first), _read(args.second))


def _readiness(args):
    from gate.s2_readiness import inspect_readiness

    request = _arguments(_read(args.request), ("plan", "evidence_index", "schedule"),
                         ("action_record",))
    return inspect_readiness(request["plan"], request["evidence_index"], request["schedule"],
                             request.get("action_record"), root=args.root)


def _report(args):
    from gate.s2_report import build_s2_report

    request = _arguments(_read(args.request), ("scope", "evidence"),
                         ("readiness", "bundle_verification", "rehearsal"))
    return build_s2_report(**request)


def _seal(args):
    from gate.s2_bundle import seal_bundle

    return seal_bundle(args.root, _read(args.index))


def _verify_bundle(args):
    from gate.s2_bundle import verify_bundle

    return verify_bundle(args.root, _read(args.manifest) if args.manifest else None)


def _prepare_full_sweep(args):
    from gate.s2_packet import build_full_sweep_packet

    return build_full_sweep_packet(_read(args.request))


# Artifact kinds that an executing path verifies with gate.s2_authority.verify_authority.
AUTHORITY_KINDS = (
    "s2_action_record", "s2_bootstrap_plan", "s4_bootstrap_plan", "s2_pilot_packet",
    "s2_full_sweep_packet", "s2_readiness_report", "s2_full_release", "s4_launch_plan",
    "s4_registration",
)


def _authority_line(args):
    """Print the exact decision-entry line that would authorize one artifact (pure).

    The hash is ``gate.s2_identity.identity_hash`` of the parsed JSON, the hash every
    executing path recomputes. Printing the line grants nothing: only Huy writes it into
    his own decision entry in docs/DECISIONS.md when he approves the artifact.
    """
    from gate.s2_identity import identity_hash

    digest = identity_hash(_read(args.artifact))
    return {
        "artifact_kind": "s2_authority_line", "schema_version": 1, "status": "passed",
        "kind": args.kind, "artifact_sha256": digest,
        "line": f"authorizes: {args.kind} sha256:{digest}",
        "grants_authority": False,
        "note": ("Only Huy writes this line, unchanged, into his own decision entry in "
                 "docs/DECISIONS.md when he approves; agents never write it."),
    }


# command -> (help, handler, acceptance key or None for status == "passed", arguments)
_REQUEST = (("request",), {})
_ROOT = (("--root",), {"type": Path, "required": True})
COMMANDS = {
    "validate-inventory": ("Validate a runtime inventory (CPU-01); no host access",
                           _validate_inventory, None, [(("inventory",), {})]),
    "compile-bootstrap": ("Compile the staged guest bootstrap plan; never runs it",
                          _compile_bootstrap, "compiled",
                          [(("inventory",), {}), (("action_record",), {}),
                           (("--phases",), {"required": True, "nargs": "+",
                                            "choices": ("A", "B", "C", "D", "E")})]),
    "describe-validity": ("Validity and complete-denominator diagnostics (CPU-04)",
                          _describe_validity, None, [_REQUEST]),
    "audit-population": ("Population/contamination audit (CPU-03)", _audit_population,
                         None, [_REQUEST]),
    "build-tensor-manifest": ("Hash safetensors tensors under --root (no model code)",
                              _build_tensor_manifest, None,
                              [_ROOT, (("--scope",), {"required": True,
                                                      "choices": ("synthetic",
                                                                  "public_source_fixture",
                                                                  "actual")}),
                               (("--index",), {"default": None}),
                               (("files",), {"nargs": "*"})]),
    "compare-tensor-manifests": ("Compare two tensor manifests", _compare_tensor_manifests,
                                 None, [(("first",), {}), (("second",), {})]),
    "describe-repeats": ("Repeatability descriptives; blocked until a method is registered",
                         _describe_repeats, None, [_REQUEST]),
    "compare-probes": (("Compare two probe artifacts whose payloads are receipted under "
                       "--root"), _compare_probes, None, [_REQUEST, _ROOT]),
    "summarize-telemetry": ("Paid-time partition, useful work and coverage (CPU-07)",
                            _summarize_telemetry, None, [_REQUEST]),
    "run-sensitivity": ("Operational power sensitivity through the protected gate",
                        _run_sensitivity, None, [_REQUEST]),
    "evaluate-registered": (("Evaluate records under a registration (synthetic or actual "
                            "scope as declared)"), _evaluate_registered, None, [_REQUEST]),
    "inspect-actual": ("Describe one unit's receipted outputs; never an admitted score",
                       _inspect_actual, None, [_REQUEST, _ROOT]),
    "readiness": ("Evidence-aware readiness sections (CPU-02); never an admission",
                  _readiness, None, [_REQUEST, _ROOT]),
    "report": ("S2 card report with separate milestone labels (CPU-13)", _report, None,
               [_REQUEST]),
    "seal": ("Seal a bundle under --root against an artifact index (CPU-11)", _seal,
             "sealed", [(("index",), {}), _ROOT]),
    "verify-bundle": ("Verify a sealed bundle under --root (CPU-11)", _verify_bundle,
                      "verified", [_ROOT, (("--manifest",), {"default": None})]),
    "prepare-full-sweep": ("Bind a registered full sweep to ordered units; never execute",
                           _prepare_full_sweep, "prepared", [_REQUEST]),
    "authority-line": (("Print the exact 'authorizes:' line for one artifact; grants "
                        "nothing"), _authority_line, None,
                       [(("--kind",), {"required": True, "choices": AUTHORITY_KINDS}),
                        (("artifact",), {})]),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight", help="Check local manifest; never authorize launch"
    )
    preflight.add_argument("manifest")
    normalize = commands.add_parser(
        "normalize", help="Admit a complete batch of local JSON records"
    )
    for name in ("summary", "identities", "observations", "policy", "observed-evaluator-sha"):
        normalize.add_argument(f"--{name}", required=True)
    normalize.add_argument("--arm", choices=("A", "B"), required=True)
    commands.add_parser("cost-table", help="Print dated GPU-only arithmetic; no spending")
    matrix = commands.add_parser("matrix", help="Print the active v5 two-arm job counts")
    matrix.add_argument("--replicates", type=int, default=3)
    schedule = commands.add_parser("schedule", help="Build explicit jobs from declared inputs")
    schedule.add_argument("request")
    ingest = commands.add_parser("ingest", help="Verify a synthetic raw-file evidence bundle")
    ingest.add_argument("request")
    ingest.add_argument("--root", type=Path, required=True)
    capture = commands.add_parser(
        "inspect-capture", help="Inspect ASL receipts/attempts without admitting scores"
    )
    capture.add_argument("request")
    capture.add_argument("--root", type=Path, required=True)
    capture.add_argument(
        "--max-asl-bytes", type=int, default=MAX_ASL_BYTES, help=MAX_ASL_BYTES_HELP,
    )
    launch = commands.add_parser(
        "prepare-launch", help="Print five-scene pilot commands; never execute them"
    )
    launch.add_argument("request")
    packet = commands.add_parser(
        "prepare-pilot", help="Bind ten pilot commands to explicit jobs/order; never execute"
    )
    packet.add_argument("request")
    for name, (text, _handler, _key, arguments) in COMMANDS.items():
        sub = commands.add_parser(name, help=text)
        for flags, options in arguments:
            sub.add_argument(*flags, **options)
    return parser


def _legacy(args):
    if args.command == "preflight":
        result = validate_s2_preflight(_read(args.manifest))
        return result, result["ready_for_planning"]
    if args.command == "normalize":
        result = normalize_summary(
            _read(args.summary),
            _read(args.identities),
            _read(args.observations),
            arm=args.arm,
            policy_id=args.policy,
            expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
            observed_evaluator_sha=args.observed_evaluator_sha,
        )
        return result, result["admitted"]
    if args.command == "matrix":
        return active_matrix_report(args.replicates), True
    if args.command == "schedule":
        request = _read(args.request)
        jobs = build_schedule(**request)
        return {
            "schema_version": 1,
            "source_request": request,
            "source_request_sha256": identity_hash(request),
            "jobs": jobs,
            "schedule_sha256": identity_hash(jobs),
            "cloud_launch_authorized": False,
        }, True
    if args.command == "ingest":
        from gate.s2_extract import extract_bundle

        result = extract_bundle(args.root, _read(args.request))
        return result, result["admitted"]
    if args.command == "inspect-capture":
        from gate.s2_capture import inspect_capture

        result = inspect_capture(
            args.root, _read(args.request), max_asl_bytes=args.max_asl_bytes
        )
        return result, result["capture_inspection_passed"]
    if args.command == "prepare-launch":
        from gate.s2_launch import build_pilot_launch

        result = build_pilot_launch(_read(args.request))
        return result, result["prepared"]
    if args.command == "prepare-pilot":
        from gate.s2_packet import build_pilot_packet

        result = build_pilot_packet(_read(args.request))
        return result, result["prepared"]
    return cost_table(), True


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command in COMMANDS:
            _text, handler, key, _arguments_spec = COMMANDS[args.command]
            result = handler(args)
            accepted = _status(result) if key is None else result.get(key) is True
        else:
            result, accepted = _legacy(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if accepted else 2
    except (OSError, ValueError, TypeError, UnicodeError, AttributeError, LookupError) as exc:
        # Review B2-07: any structural defect in an input prints the documented error
        # JSON (exit 2) rather than a traceback.
        print(json.dumps({"error": str(exc) or type(exc).__name__, "spend_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
