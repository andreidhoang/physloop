"""Create fresh append-only evidence for synthetic S2 ingestion qualification.

The fixture has invented outcomes and a PhysLoop synthetic execution trace.
Passing this verifier does not qualify AlpaSim runtime capture or a scientific run.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "evidence/s1-20261031-autonomous-v1/manifest.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from s2_ingestion_fixture import fixture_bundle, metric_rows, parquet_receipt

from gate.s2_extract import extract_bundle
from gate.s2_identity import read_json_bytes
from gate.s2_preflight import validate_s2_preflight
from gate.s2_schedule import active_matrix_report


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_demo(output):
    """Create a new fixture tree and verify literal known answers, without subprocesses."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    positive_root, negative_root = output / "synthetic", output / "missing-offroad"
    positive_root.mkdir()
    negative_root.mkdir()
    positive_request, _, _, _ = fixture_bundle(positive_root)
    write_json(positive_root / "request.json", positive_request)
    positive = extract_bundle(positive_root, positive_request)
    write_json(positive_root / "result.json", positive)

    negative_request, _, _, _ = fixture_bundle(negative_root)
    filtered = negative_root / "filtered"
    filtered.mkdir()
    rows = [row for row in metric_rows() if row["name"] != "offroad"]
    negative_receipt = parquet_receipt(filtered, rows)
    negative_receipt["path"] = "filtered/metrics.parquet"
    negative_request["artifacts"][0]["metrics"] = negative_receipt
    write_json(negative_root / "request.json", negative_request)
    negative = extract_bundle(negative_root, negative_request)
    write_json(negative_root / "result.json", negative)

    records = [
        record for batch in positive["batches"] for record in batch["normalization"]["records"]
    ]
    score = records[0]["value"] if len(records) == 1 else None
    deviation = records[0]["deviation_m"] if len(records) == 1 else None
    outside = sum(
        batch["normalization"]["counts"]["out_of_envelope"] for batch in positive["batches"]
    )
    negative_codes = sorted({issue["code"] for issue in negative["issues"]})
    positive_ok = (
        positive["admitted"]
        and len(records) == 1
        and score == 0.8
        and deviation == 4.2
        and outside == 1
    )
    negative_ok = (
        not negative["admitted"]
        and "missing_raw_metric" in negative_codes
        and all(not batch["normalization"]["records"] for batch in negative["batches"])
    )
    drafts = {}
    for name in ("S2_PILOT_DRAFT", "S2_FULL_DRAFT"):
        draft = read_json_bytes((ROOT / f"experiments/{name}.json").read_bytes())
        result = validate_s2_preflight(draft)
        write_json(output / f"{name.lower()}_preflight.json", result)
        drafts[name] = {
            "ready": result["ready_for_planning"],
            "blocker_count": len(result["blocking_reasons"]),
        }
    write_json(output / "active_matrix.json", active_matrix_report())
    runtime_unqualified = all(
        result["runtime_qualified"] is False and result["cloud_launch_authorized"] is False
        for result in (positive, negative)
    )
    demo = {
        "scope": "synthetic known answers only; full checks are separate",
        "passed": positive_ok
        and negative_ok
        and runtime_unqualified
        and all(not result["ready"] for result in drafts.values()),
        "positive": {
            "admitted": positive["admitted"],
            "score": score,
            "deviation_m": deviation,
            "out_of_envelope": outside,
            "known_answer_passed": positive_ok,
        },
        "missing_offroad": {
            "admitted": negative["admitted"],
            "issue_codes": negative_codes,
            "known_answer_passed": negative_ok,
        },
        "drafts": drafts,
        "runtime_qualified": False,
        "runtime_limit_preserved": runtime_unqualified,
        "full_checks_run": False,
        "cloud_launch_authorized": False,
    }
    write_json(output / "demo.json", demo)
    return demo


def source_snapshot():
    """Snapshot explicit source trees, never host logs or other evidence directories."""
    files = {
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / BASELINE,
        ROOT / ".claude/starred-cores.txt",
        ROOT / "docs/02_PHYSLOOP_PLAN.md",
        ROOT / "docs/research/2026-11-07-pilot-launch-audit.md",
    }
    for directory in ("gate", "tests", "scripts", "experiments"):
        files.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    files.update((ROOT / "docs").glob("S2*.md"))
    snapshot = {}
    for path in sorted(files):
        if path.is_symlink():
            raise ValueError(f"Source archive refuses symlink: {path.relative_to(ROOT)}")
        snapshot[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return snapshot


def source_hashes(snapshot):
    return {name: hashlib.sha256(data).hexdigest() for name, data in snapshot.items()}


def protected_sources_unchanged(snapshot):
    baseline = read_json_bytes(snapshot[BASELINE])["source_sha256"]
    protected = {
        name: digest
        for name, digest in baseline.items()
        if name == "gate/verdict.py" or name.startswith("gate/stats/")
    }
    observed = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in snapshot.items()
        if name == "gate/verdict.py" or name.startswith("gate/stats/")
    }
    return bool(protected) and observed == protected


def check_locked_versions(lock, installed):
    """Check direct runtime/dev version identity; not a substitute for clean sync."""
    packages = {package["name"]: package for package in lock["package"]}
    project = packages["physloop-gate"]
    runtime = {item["name"] for item in project.get("dependencies", [])}
    dependencies = runtime | {
        item["name"] for group in project.get("dev-dependencies", {}).values() for item in group
    }
    comparisons = {
        name: {"locked": packages.get(name, {}).get("version"), "installed": installed.get(name)}
        for name in sorted(dependencies)
    }
    return {
        "matched": "pyarrow" in runtime
        and all(
            item["locked"] is not None and item["locked"] == item["installed"]
            for item in comparisons.values()
        ),
        "packages": comparisons,
        "clean_frozen_sync_verified": False,
    }


def environment_record(snapshot):
    lock = tomllib.loads(snapshot["uv.lock"].decode())
    installed = {}
    for package in lock["package"]:
        try:
            installed[package["name"]] = version(package["name"])
        except PackageNotFoundError:
            installed[package["name"]] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **check_locked_versions(lock, installed),
    }


def run_checks(output):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    checks = []
    for name, command in (
        ("tests", [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"]),
        (
            "lint",
            [sys.executable, "-B", "-m", "ruff", "check", "--no-cache", "gate", "tests", "scripts"],
        ),
        ("diff_whitespace", ["git", "diff", "--check"]),
    ):
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
            returncode, log = result.returncode, result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            returncode, log = -1, f"Check failed to finish: {exc}\n"
        with (output / f"{name}.log").open("x", encoding="utf-8") as stream:
            stream.write(log)
        checks.append({"name": name, "command": command, "returncode": returncode})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before = source_snapshot()
    hashes = source_hashes(before)
    environment = environment_record(before)
    demo = run_demo(args.output)
    output = args.output.resolve(strict=True)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in before.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    bundle_hash = hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()
    checks = run_checks(output)
    source_recheck_error = None
    try:
        after = source_snapshot()
        unchanged = source_hashes(after) == hashes
        protected = protected_sources_unchanged(before) and protected_sources_unchanged(after)
    except (OSError, ValueError) as exc:
        unchanged, protected = False, False
        source_recheck_error = str(exc)
    passed = demo["passed"] and all(item["returncode"] == 0 for item in checks)
    passed = passed and unchanged and protected and environment["matched"]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "schema_version": 1,
        "scope": "S2 synthetic raw-file ingestion only",
        "created_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "runtime_qualified": False,
        "cloud_launch_authorized": False,
        "cloud_spend_usd": 0,
        "gpu_runs": 0,
        "licensed_assets_downloaded": False,
        "checks": checks,
        "demo": demo,
        "full_checks_run": True,
        "environment": environment,
        "source_sha256": hashes,
        "source_bundle_sha256": bundle_hash,
        "sources_unchanged_during_checks": unchanged,
        "source_recheck_error": source_recheck_error,
        "s1_protected_sources_unchanged": protected,
        "git_base": git.stdout.strip() if git.returncode == 0 else None,
        "unverified": [
            "clean frozen environment rebuild (installed direct versions checked against lock)",
            "real ASL/runtime capture and terminal semantics",
            "physical validity and authenticity of observations",
            "registered population, scientific definitions and sampling assumptions",
            "full scorer recomputation and model/renderer reproducibility",
            "GPU performance, reliability, rights and concrete execution authority",
        ],
    }
    write_json(output / "verification.json", report)
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(
            f"Synthetic S2 ingestion verification: **{'PASS' if passed else 'FAIL'}**.\n\n"
            f"Known-answer score: {demo['positive']['score']}; full deviation: "
            f"{demo['positive']['deviation_m']} m; missing offroad rejected: "
            f"{demo['missing_offroad']['known_answer_passed']}. Both scientific drafts "
            "must remain blocked.\n\n"
            f"Full checks passed: {all(item['returncode'] == 0 for item in checks)}; "
            f"sources unchanged during checks: {unchanged}; S1 protected sources "
            f"unchanged: {protected}.\n\n"
            f"Source bundle SHA-256: `{bundle_hash}`. `verification.json` contains "
            "per-file hashes and exact commands; each check has its captured log.\n\n"
            "The inputs and execution trace are invented CPU fixtures. Runtime "
            "qualification remains false. No simulator, model, GPU or licensed asset "
            "was exercised, and no scientific admission or launch authority follows.\n"
        )
    write_json(
        output / "checksums.json",
        {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps({"passed": passed, "evidence": str(output), "runtime_qualified": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
