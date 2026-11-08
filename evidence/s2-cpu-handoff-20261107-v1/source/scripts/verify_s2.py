"""Generate a fresh, append-only record of the offline S2 preparation checks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate.s2 import cost_table
from gate.s2_preflight import validate_s2_preflight


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "evidence/s1-20261031-autonomous-v1/manifest.json",
    ]
    for directory in ("gate", "tests", "scripts", "experiments"):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    for name in (
        "S2_ENGINEERING_SPEC.md",
        "S2_IMPLEMENTATION_PLAN.md",
        "RESEARCH_OPERATING_MODEL.md",
        "DECISIONS.md",
        "00_START_HERE.md",
    ):
        paths.append(ROOT / "docs" / name)
    paths.extend((ROOT / "docs/research").glob("2026-10-31-s2-*.md"))
    source_bytes = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(set(paths))}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()}
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, value in source_bytes.items():
            archive.writestr(name, value)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    checks = []
    for name, command in (
        ("tests", [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"]),
        ("lint", [sys.executable, "-B", "-m", "ruff", "check", "gate", "tests", "scripts"]),
        ("diff_whitespace", ["git", "diff", "--check"]),
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        with (output / f"{name}.log").open("x") as stream:
            stream.write(result.stdout + result.stderr)
        checks.append({"name": name, "command": command, "returncode": result.returncode})
    drafts = {}
    for name in ("S2_PILOT_DRAFT", "S2_FULL_DRAFT"):
        manifest = json.loads((ROOT / f"experiments/{name}.json").read_text())
        report = validate_s2_preflight(manifest)
        write_json(output / f"{name.lower()}_preflight.json", report)
        drafts[name] = {
            "ready": report["ready_for_planning"],
            "blocker_count": len(report["blocking_reasons"]),
        }
    write_json(output / "cost_table.json", cost_table())
    prior = json.loads((ROOT / "evidence/s1-20261031-autonomous-v1/manifest.json").read_text())
    protected_unchanged = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in prior["source_sha256"].items()
        if name == "gate/verdict.py" or name.startswith("gate/stats/")
    )
    unchanged_during_checks = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )
    passed = (
        all(c["returncode"] == 0 for c in checks)
        and protected_unchanged
        and unchanged_during_checks
        and all(not d["ready"] for d in drafts.values())
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
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
