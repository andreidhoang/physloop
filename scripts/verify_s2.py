"""Generate a fresh, append-only record of the offline S2 preparation checks.

Tickets: R4 C12. Sources are read once into a frozen snapshot; the whitespace check scans that
snapshot (staged, unstaged and untracked files alike, hash-verified vendored upstream files
exempt) instead of ``git diff --check``; Ruff runs with ``--no-cache``; the protected-source
check compares the whole set of ``gate/verdict.py`` and ``gate/stats/`` files against the S1
baseline, so an added, removed or changed file fails it; drafts are parsed from the snapshot
with the strict reader. Verification level: L1 for ``source_snapshot``, ``draft_reports`` and
``run_checks`` with an injected runner; ``main`` runs the live suite, which checks the current
checkout and does not qualify a later state. Never: commits, downloads, rents, edits existing
evidence, or authorizes a launch.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate.s2 import cost_table
from gate.s2_identity import read_json_bytes
from gate.s2_preflight import validate_s2_preflight
from scripts.verify_s2_ingestion import (
    BASELINE,
    check_environment,
    protected_sources_unchanged,
    run_checks,
    source_hashes,
)

DRAFTS = ("S2_PILOT_DRAFT", "S2_FULL_DRAFT")


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_snapshot(root=None):
    """Read the S2 preparation sources once; refuse symlinks; keys are POSIX relative paths."""
    root = ROOT if root is None else Path(root)
    paths = [root / "pyproject.toml", root / "uv.lock", root / BASELINE]
    for directory in ("gate", "teacher_rl", "tests", "scripts", "experiments"):
        paths.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    for name in (
        "S2_ENGINEERING_SPEC.md",
        "S2_IMPLEMENTATION_PLAN.md",
        "RESEARCH_OPERATING_MODEL.md",
        "DECISIONS.md",
        "00_START_HERE.md",
    ):
        paths.append(root / "docs" / name)
    paths.extend((root / "docs/research").glob("2026-10-31-s2-*.md"))
    snapshot = {}
    for path in sorted(set(paths)):
        if path.is_symlink():
            raise ValueError(f"Source archive refuses symlink: {path.relative_to(root)}")
        snapshot[path.relative_to(root).as_posix()] = path.read_bytes()
    return snapshot


def draft_reports(snapshot):
    """Preflight each scientific draft from the frozen snapshot with the strict reader."""
    reports = {}
    for name in DRAFTS:
        manifest = read_json_bytes(snapshot[f"experiments/{name}.json"])
        reports[name] = validate_s2_preflight(manifest)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = source_snapshot()
    hashes = source_hashes(snapshot)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, value in snapshot.items():
            archive.writestr(name, value)
    checks = run_checks(output, snapshot)
    drafts = {}
    for name, report in draft_reports(snapshot).items():
        write_json(output / f"{name.lower()}_preflight.json", report)
        drafts[name] = {
            "ready": report["ready_for_planning"],
            "blocker_count": len(report["blocking_reasons"]),
        }
    write_json(output / "cost_table.json", cost_table())
    protected_unchanged = protected_sources_unchanged(snapshot)
    try:
        after = source_snapshot()
        unchanged_during_checks = source_hashes(after) == hashes
        protected_unchanged = protected_unchanged and protected_sources_unchanged(after)
    except (OSError, ValueError):
        unchanged_during_checks, protected_unchanged = False, False
    passed = (
        all(c["returncode"] == 0 for c in checks)
        and protected_unchanged
        and unchanged_during_checks
        and all(not d["ready"] for d in drafts.values())
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        env=check_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "schema_version": 1,
        "scope": "S2 offline preparation only",
        "created_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "cloud_spend_usd": 0,
        "gpu_runs": 0,
        "licensed_assets_downloaded": False,
        "cloud_launch_authorized": False,
        "checks": checks,
        "drafts": drafts,
        "s1_protected_sources_unchanged": protected_unchanged,
        "sources_unchanged_during_checks": unchanged_during_checks,
        "git_base": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": hashes,
        "unverified": [
            "real AlpaSim integration",
            "sidecar authenticity",
            "road sampling assumptions",
            "full evaluator/runtime",
            "GPU performance/reliability",
            "license/budget decisions",
        ],
    }
    write_json(output / "verification.json", report)
    write_json(
        output / "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    )
    print(json.dumps({"passed": passed, "evidence": str(output), "drafts": drafts}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
