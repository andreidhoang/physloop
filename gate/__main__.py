"""CPU-only command line: verdict, power, and preregistered synthetic qualification.

Tickets: R4 C3 (the verdict request is parsed with the strict reader
``gate.s2_identity.read_json_bytes``, so duplicate keys, nonfinite constants and invalid UTF-8
are rejected instead of silently resolved). Verification level: L1 (in-process and fresh
interpreter CLI tests with literal known answers). Never: launches, downloads, authorizes, or
recomputes the protected bound; on an input error it exits 2 with an error JSON on stdout and an
``input or execution error`` line on stderr, and never prints a verdict.
"""

import argparse
import json
import sys
from pathlib import Path

from gate.api import evaluate, parse_request
from gate.s2_identity import read_json_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physloop-gate")
    commands = parser.add_subparsers(dest="command", required=True)
    decision = commands.add_parser("verdict", help="admit raw paired arms and produce a verdict")
    decision.add_argument("input", type=Path)
    commands.add_parser("power", help="print labelled planning and operational tables as JSON")
    experiment = commands.add_parser(
        "qualify", help="generate a new append-only synthetic evidence bundle"
    )
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--trials", type=int, default=1000)
    experiment.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)
    try:
        if args.command == "verdict":
            request = read_json_bytes(args.input.read_bytes())
            records, spec, candidate_protocol = parse_request(request)
            result = evaluate(records, spec, candidate_protocol=candidate_protocol)
        elif args.command == "power":
            from gate.power import power_tables

            result = power_tables()
        else:
            from gate.experiment import run_experiment

            result = run_experiment(args.output, args.trials, args.seed)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        # Valid NO-GO/INSUFFICIENT are results, not process failures.
        if args.command == "qualify" and not result["experiment_pass"]:
            return 1
        return 0
    except (ValueError, TypeError, OSError, KeyError) as exc:
        error = {
            "schema_version": 1,
            "command": args.command,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "verdict_emitted": False,
        }
        print(json.dumps(error, indent=2, sort_keys=True))
        print(f"input or execution error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
