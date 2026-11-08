"""Offline S2 tooling. Does not provision, download assets, or launch experiments."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA, normalize_summary
from gate.s2_identity import identity_hash, read_json_bytes
from gate.s2_preflight import validate_s2_preflight
from gate.s2_schedule import active_matrix_report, build_schedule


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--max-asl-bytes", type=int, default=256 * 1024 * 1024,
        help="Operator memory bound per ASL file (default 256 MiB); bytes are read into RAM",
    )
    launch = commands.add_parser(
        "prepare-launch", help="Print five-scene pilot commands; never execute them"
    )
    launch.add_argument("request")
    packet = commands.add_parser(
        "prepare-pilot", help="Bind ten pilot commands to explicit jobs/order; never execute"
    )
    packet.add_argument("request")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = validate_s2_preflight(_read(args.manifest))
            accepted = result["ready_for_planning"]
        elif args.command == "normalize":
            result = normalize_summary(
                _read(args.summary),
                _read(args.identities),
                _read(args.observations),
                arm=args.arm,
                policy_id=args.policy,
                expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
                observed_evaluator_sha=args.observed_evaluator_sha,
            )
            accepted = result["admitted"]
        elif args.command == "matrix":
            result, accepted = active_matrix_report(args.replicates), True
        elif args.command == "schedule":
            request = _read(args.request)
            jobs = build_schedule(**request)
            result = {
                "schema_version": 1,
                "source_request": request,
                "source_request_sha256": identity_hash(request),
                "jobs": jobs,
                "schedule_sha256": identity_hash(jobs),
                "cloud_launch_authorized": False,
            }
            accepted = True
        elif args.command == "ingest":
            from gate.s2_extract import extract_bundle

            result = extract_bundle(args.root, _read(args.request))
            accepted = result["admitted"]
        elif args.command == "inspect-capture":
            from gate.s2_capture import inspect_capture

            result = inspect_capture(
                args.root, _read(args.request), max_asl_bytes=args.max_asl_bytes
            )
            accepted = result["capture_inspection_passed"]
        elif args.command == "prepare-launch":
            from gate.s2_launch import build_pilot_launch

            result = build_pilot_launch(_read(args.request))
            accepted = result["prepared"]
        elif args.command == "prepare-pilot":
            from gate.s2_packet import build_pilot_packet

            result = build_pilot_packet(_read(args.request))
            accepted = result["prepared"]
        else:
            result, accepted = cost_table(), True
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if accepted else 2
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc), "spend_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
