"""CPU-only command line: verdict, power, and preregistered synthetic qualification."""

import argparse
import json
import sys
from pathlib import Path

from gate.api import evaluate, parse_request


def main():
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
    args = parser.parse_args()
    try:
        if args.command == "verdict":
            records, spec, candidate_protocol = parse_request(json.loads(args.input.read_text()))
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
        print(f"input or execution error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
