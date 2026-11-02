"""Post-run independent aggregation, artifact, reproduction and rare-count audit.

Run from a source checkout. Writes a NEW evidence directory, never edits a run.
The vectorized bisection below intentionally does not call gate's bound functions.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import binom


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_upper(mean, n, tail):
    left, right = mean.copy(), np.ones_like(mean)
    target = -math.log(tail) / n
    for _ in range(100):
        q = (left + right) / 2
        kl = xlogy(mean, mean) - xlogy(mean, q) + xlogy(1 - mean, 1 - mean) - xlogy(1 - mean, 1 - q)
        inside = kl <= target
        left = np.where(inside, q, left)
        right = np.where(inside, right, q)
    return right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--wheel-smoke", type=Path, required=True)
    args = parser.parse_args()
    primary, reproduction, output = args.primary, args.reproduction, args.output
    if output.exists():
        raise FileExistsError("verification directory already exists")
    output.mkdir(parents=True)
    checksums = json.loads((primary / "checksums.json").read_text())
    assert all(digest(primary / name) == expected for name, expected in checksums.items())
    manifest = json.loads((primary / "manifest.json").read_text())
    assert digest(primary / "config.json") == manifest["config_sha256"]
    with zipfile.ZipFile(primary / "source.zip") as archive:
        assert set(archive.namelist()) == set(manifest["source_sha256"])
        for name, expected in manifest["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    matches = {}
    for name in ("trials.jsonl", "config.json", "power_tables.json", "outcomes.svg"):
        matches[name] = digest(primary / name) == digest(reproduction / name)
    assert all(matches.values())
    summary = json.loads((primary / "summary.json").read_text())
    repeated = json.loads((reproduction / "summary.json").read_text())
    assert summary["cells"] == repeated["cells"]
    counts = defaultdict(Counter)
    seeds = set()
    records = 0
    with (primary / "trials.jsonl").open() as stream:
        for line in stream:
            trial = json.loads(line)
            records += 1
            seeds.add((trial["seed"], trial["order_seed"]))
            for mode, result in trial["results"].items():
                counts[f"{trial['case']}/{mode}"][result["overall"]] += 1
    assert records == 10000 and len(seeds) == records
    for name, count in counts.items():
        assert dict(count) == summary["cells"][name]["counts"]
    # Post hoc enumeration of the unexpected three boundary errors, not a new calibration rule.
    n = 220
    plus, minus = np.array([(i, j) for i in range(n + 1) for j in range(n - i + 1)]).T
    zero = n - plus - minus
    upper = independent_upper(plus / n, n, 0.025) + 1e-12
    lower_negative = 1 - independent_upper(1 - minus / n, n, 0.025) - 1e-12
    go = upper - np.maximum(0, lower_negative) < 0.02
    probability = np.exp(
        gammaln(n + 1)
        - gammaln(plus + 1)
        - gammaln(minus + 1)
        - gammaln(zero + 1)
        + plus * np.log(0.04)
        + minus * np.log(0.02)
        + zero * np.log(0.94)
    )
    exact_go = float(probability[go].sum())
    exact_diagnostic = {
        "classification": "post_hoc_diagnostic_not_changed_acceptance",
        "count_states": len(plus),
        "false_go_probability": exact_go,
        "expected_errors_per_1000": 1000 * exact_go,
        "probability_at_least_three_in_1000": float(binom.sf(2, 1000, exact_go)),
        "conclusion": "unusual count retained; source/data replay and aggregate audit found no defect",
    }
    commands = [
        [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"],
        [str(Path(sys.executable).parent / "ruff"), "check", "gate", "tests"],
    ]
    test_results = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (output / f"check-{index}.txt").write_text(result.stdout + result.stderr)
        test_results.append({"command": command, "returncode": result.returncode})
        assert result.returncode == 0, result.stdout + result.stderr
    for artifact in args.wheel_dir.glob("physloop_gate-*"):
        shutil.copyfile(artifact, output / artifact.name)
    smoke = json.loads(args.wheel_smoke.read_text())
    assert smoke["result"] == "GO" and smoke["exit_code"] == 0
    result = {
        "primary_manifest_sha256": digest(primary / "manifest.json"),
        "primary_artifacts_verified": len(checksums),
        "source_files_verified": len(manifest["source_sha256"]),
        "trial_records": records,
        "reproduction_bitwise_matches": matches,
        "regression_counts_recomputed": True,
        "exact_posthoc": exact_diagnostic,
        "checks": test_results,
        "installed_wheel_smoke": smoke,
        "note": "Independent audit does not establish sampling assumptions for AlpaSim.",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(__file__, output / "verify_s1.py")
    out_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(out_hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
