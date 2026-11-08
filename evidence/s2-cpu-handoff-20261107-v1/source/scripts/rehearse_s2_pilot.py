"""Rehearse the minimal S2 pilot on CPU from a frozen source copy.

No provider, asset, Docker or model calls. The existing watchdog runs against a
fake inventory; ten jobs use the existing synthetic capture/ingestion contract.
Exit 2 preserves an unresolved environment lock even when CPU checks pass.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tomllib
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from s2_capture_fixture import wire_capture_request
from s2_ingestion_fixture import receipt
from s2_launch_fixture import launch_request
from s2_pilot_fixture import build_pilot_fixture

from gate.s2_capture import inspect_capture
from gate.s2_extract import extract_bundle
from gate.s2_identity import read_json_bytes
from gate.s2_launch import build_pilot_launch
from gate.s2_pilot import copy_verified_export, next_unit_check, verify_export
from gate.s2_preflight import validate_s2_preflight
from scripts import vast_stop_watchdog as watchdog
from scripts.verify_s2_ingestion import (
    environment_record,
    protected_sources_unchanged,
    source_hashes,
    source_snapshot,
    write_json,
)


def receipts_for(root):
    return [
        {"path": path.relative_to(root).as_posix(),
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def watchdog_demo(output):
    """Exercise the real stop predicate with fake time and provider responses."""
    class Clock:
        wall = 100.0

        def time(self):
            return self.wall

        def monotonic(self):
            return self.wall - 100

        def sleep(self, seconds):
            self.wall += seconds

    running = {"id": 123, "label": "synthetic-pilot", "actual_status": "running",
               "intended_status": "running", "next_state": "running"}
    stopped = {**running, "actual_status": "exited", "intended_status": "stopped",
               "next_state": "stopped"}
    responses = iter([[running], [running], {"success": True}, [stopped]])
    calls = []

    def fake_call(cli, args, timeout):
        calls.append(args)
        return next(responses)

    with (
        patch.object(watchdog, "_call", fake_call),
        redirect_stdout(io.StringIO()),
        watchdog.EventLog(output / "watchdog-events.jsonl") as log,
    ):
        result = watchdog.watch(
            123, "synthetic-pilot", 101, log, clock=Clock(), poll_seconds=1
        )
    events = [json.loads(line) for line in (output / "watchdog-events.jsonl").read_text().splitlines()]
    return result == 0 and calls == [
        ["show", "instances"], ["show", "instances"],
        ["stop", "instance", "123"], ["show", "instances"],
    ] and events[0]["event"] == "armed" and events[-1]["event"] == "stopped_verified"


def run_demo(output):
    """Cheap fixture run; unit tests call this without recursively launching tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    raw = output / "synthetic"
    raw.mkdir()
    request = build_pilot_fixture(raw)
    positive = extract_bundle(raw, request)
    write_json(raw / "result.json", positive)
    negative_request = copy.deepcopy(request)
    negative_request["artifacts"].pop()
    negative = extract_bundle(raw, negative_request)
    write_json(output / "missing-last-job.json", negative)
    receipts = receipts_for(raw)
    exported = copy_verified_export(raw, output / "export", receipts)
    write_json(output / "export-manifest.json", receipts)
    replay = extract_bundle(output / "export", request)

    corrupt = output / "corrupt-export-control"
    corrupt.mkdir()
    first = receipts[0]
    bad_file = corrupt / first["path"]
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_bytes(b"deliberately corrupted synthetic export")
    corrupt_result = verify_export(corrupt, [first])
    write_json(output / "corrupt-export-result.json", corrupt_result)

    budget = {
        "scopes": {name: {"cap_usd": "10", "used_usd": "6", "liability_usd": "1",
                          "reserve_usd": "1"} for name in ("pilot", "stage", "project")},
        "next_unit_usd": "1.99", "now_epoch": 100, "deadline_epoch": 200,
        "next_unit_seconds": 60, "cleanup_seconds": 20,
    }
    affordable = next_unit_check(budget)
    exhausted = next_unit_check({**budget, "next_unit_usd": "2"})
    expired = next_unit_check({**budget, "now_epoch": 120})
    write_json(output / "budget-controls.json", {
        "synthetic_inputs": budget, "affordable": affordable,
        "exhausted": exhausted, "deadline": expired,
    })
    draft = validate_s2_preflight(read_json_bytes(
        (ROOT / "experiments/S2_PILOT_DRAFT.json").read_bytes()
    ))
    write_json(output / "scientific-preflight.json", draft)
    setup = read_json_bytes((ROOT / "experiments/S2_SETUP_DRAFT.json").read_bytes())
    write_json(output / "setup-draft.json", setup)
    capture_root = output / "synthetic-asl"
    capture_root.mkdir()
    capture_request = wire_capture_request(capture_root)
    capture = inspect_capture(capture_root, capture_request)
    write_json(capture_root / "request.json", capture_request)
    write_json(capture_root / "result.json", capture)
    torn_request = copy.deepcopy(capture_request)
    torn_request["artifacts"][0]["asl"] = receipt(
        capture_root, "torn.asl", (capture_root / "capture.asl").read_bytes()[:-1]
    )
    torn = inspect_capture(capture_root, torn_request)
    write_json(capture_root / "torn-request.json", torn_request)
    write_json(capture_root / "torn-result.json", torn)
    decoded = capture["captures"][0]["inspection"]
    torn_decoded = torn["captures"][0]["inspection"]
    asl_checks = {
        "passed": capture["capture_inspection_passed"],
        "pose_events": decoded["observations"]["pose_event_count"],
        "actual_steps": decoded["actual_steps"],
        "torn_frame_rejected": not torn["capture_inspection_passed"] and
            "truncated_frame" in {issue["code"] for issue in torn_decoded["issues"]},
        "score_admitted": capture["score_admitted"],
        "runtime_qualified": capture["runtime_qualified"],
    }
    launch_input = launch_request()
    launch = build_pilot_launch(launch_input)
    incomplete_launch = build_pilot_launch({**launch_input, "scene_ids": []})
    write_json(output / "synthetic-launch-request.json", launch_input)
    write_json(output / "synthetic-launch-result.json", launch)
    write_json(output / "missing-scenes-result.json", incomplete_launch)
    launch_checks = {
        "prepared": launch["prepared"],
        "policy_rollouts": launch["pilot"]["policy_rollouts"],
        "missing_scenes_rejected": not incomplete_launch["prepared"],
        "commands_executed": launch["commands_executed"],
    }
    records = [row for batch in positive["batches"]
               for row in batch["normalization"]["records"]]
    outside = sum(batch["normalization"]["counts"]["out_of_envelope"]
                  for batch in positive["batches"])
    negative_controls = {
        "missing_last_artifact": not negative["admitted"] and
        "missing_attempt_artifacts" in {issue["code"] for issue in negative["issues"]} and all(
            not batch["normalization"]["records"] for batch in negative["batches"]),
        "export_corruption": not corrupt_result["verified"],
        "budget_boundary": not exhausted["can_start_next_unit"],
        "deadline_boundary": not expired["can_start_next_unit"],
    }
    watchdog_passed = watchdog_demo(output)
    report = {
        "scope": "synthetic ten-job pilot; no actual AlpaSim or provider execution",
        "jobs": len(request["schedule"]), "records": len(records),
        "scores": [row["value"] for row in records],
        "deviations_m": [row["deviation_m"] for row in records],
        "out_of_envelope": outside, "negative_controls": negative_controls,
        "export_verified": exported["verified"] and replay == positive,
        "fake_watchdog_passed": watchdog_passed,
        "scientific_draft_ready": draft["ready_for_planning"],
        "scientific_blockers": draft["blocking_reasons"],
        "asl_capture": asl_checks,
        "launch_preparation": launch_checks,
        "runtime_qualified": False, "cloud_launch_authorized": False,
    }
    report["passed"] = (
        positive["admitted"] and report["jobs"] == report["records"] == outside == 10
        and report["scores"] == [0.8] * 10 and report["deviations_m"] == [4.2] * 10
        and all(negative_controls.values()) and report["export_verified"]
        and affordable["can_start_next_unit"] and watchdog_passed
        and not draft["ready_for_planning"]
        and setup["cloud_launch_authorized"] is False and setup["status"] == "draft"
        and asl_checks["passed"] and asl_checks["pose_events"] == 4
        and asl_checks["actual_steps"] is None and asl_checks["torn_frame_rejected"]
        and not asl_checks["score_admitted"] and not asl_checks["runtime_qualified"]
        and launch_checks["prepared"] and launch_checks["policy_rollouts"] == 10
        and launch_checks["missing_scenes_rejected"] and not launch_checks["commands_executed"]
    )
    write_json(output / "demo.json", report)
    return report


def declaration_check(snapshot):
    """Check this repository's simple requirement declarations against lock metadata.

    Unsupported requirement forms block instead of guessing PEP 508 semantics.
    This consistency check is not a resolver or a clean frozen installation.
    """
    issues = []
    try:
        project = tomllib.loads(snapshot["pyproject.toml"].decode())
        lock = tomllib.loads(snapshot["uv.lock"].decode())
        root = next(item for item in lock["package"]
                    if item["name"] == project["project"]["name"])

        def requirement(value):
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)(==[A-Za-z0-9.*+-]+)?", value)
            if not match:
                raise ValueError("Unsupported requirement form")
            return re.sub(r"[-_.]+", "-", match[1]).lower(), match[2] or ""

        def locked_requirements(items):
            if any(set(item) - {"name", "specifier"} for item in items):
                raise ValueError("Unsupported lock requirement form")
            return sorted(requirement(item["name"] + item.get("specifier", ""))
                          for item in items)

        declared = sorted(requirement(item) for item in project["project"]["dependencies"])
        metadata = root["metadata"]
        if declared != locked_requirements(metadata["requires-dist"]):
            issues.append("runtime_declarations_differ_from_lock")
        if {name for name, _ in declared} != {item["name"] for item in root["dependencies"]}:
            issues.append("runtime_dependency_set_differs_from_lock")
        groups = project.get("dependency-groups", {})
        locked_groups = metadata.get("requires-dev", {})
        resolved_groups = root.get("dev-dependencies", {})
        if set(groups) != set(locked_groups) or set(groups) != set(resolved_groups):
            issues.append("dev_group_set_differs_from_lock")
        for name, requirements in groups.items():
            expected = sorted(requirement(item) for item in requirements)
            if expected != locked_requirements(
                locked_groups.get(name, [])
            ):
                issues.append(f"dev_declarations_differ_from_lock:{name}")
            if {item[0] for item in expected} != {
                requirement(item["name"])[0] for item in resolved_groups.get(name, [])
            }:
                issues.append(f"dev_dependency_set_differs_from_lock:{name}")
        python = project["project"]["requires-python"]
        if python != lock["requires-python"] or python != "==3.12.*" or sys.version_info[:2] != (3, 12):
            issues.append("python_constraint_unmatched_or_unsupported")
    except (KeyError, ValueError, TypeError, StopIteration):
        issues.append("unsupported_or_incomplete_environment_declaration")
    return {"matched": not issues, "issues": issues, "frozen_sync_verified": False}


def run_frozen_checks(output):
    """Snapshot once; execute fixture, suite and lint from those exact bytes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    snapshot = source_snapshot()
    hashes = source_hashes(snapshot)
    frozen = output / "source"
    frozen.mkdir()
    for name, data in snapshot.items():
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    write_json(output / "source-manifest.json", hashes)
    environment = environment_record(snapshot)
    environment["declarations"] = declaration_check(snapshot)
    environment["matched"] = environment["matched"] and environment["declarations"]["matched"]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
        timeout=10, check=False,
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(frozen)}
    checks = []
    commands = (
        ("demo", [sys.executable, "-B", "scripts/rehearse_s2_pilot.py", "--demo-only",
                  "--output", str(output / "demo")]),
        ("tests", [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"]),
        ("lint", [sys.executable, "-B", "-m", "ruff", "check", "--no-cache",
                  "gate", "tests", "scripts"]),
    )
    for name, command in commands:
        try:
            result = subprocess.run(command, cwd=frozen, env=env, capture_output=True,
                                    text=True, timeout=900, check=False)
            code, log = result.returncode, result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired):
            code, log = -1, "CPU check could not finish.\n"
        with (output / f"{name}.log").open("x") as stream:
            stream.write(log)
        checks.append({"name": name, "argv": command, "returncode": code})
    unchanged = all(hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest
                    for name, digest in hashes.items())
    protected = protected_sources_unchanged(snapshot)
    cpu_passed = all(check["returncode"] == 0 for check in checks) and unchanged and protected
    report = {
        "scope": "minimal S2 CPU preparation only",
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "source_identity": "included source-manifest.json; includes uncommitted changes",
        "cpu_checks_passed": cpu_passed,
        "environment": environment,
        "cpu_handoff_ready": cpu_passed and environment["matched"],
        "cpu_handoff_ready_scope": "current interpreter and frozen source; clean rebuild separate",
        "clean_frozen_sync_verified": environment["clean_frozen_sync_verified"],
        "checks_executed_from_snapshot": True,
        "snapshot_unchanged": unchanged, "protected_sources_unchanged": protected,
        "checks": checks, "runtime_qualified": False, "cloud_launch_authorized": False,
        "cloud_spend_usd": 0, "gpu_runs": 0,
        "remaining": [
            "fresh frozen environment sync (installed direct versions only checked here)",
            "five real scenes, exact pilot launcher and runtime/asset identities",
            "owner scientific registration, rights/access and concrete rental scope",
            "actual AlpaSim capture/scorer integration and host/model/renderer fit",
        ],
    }
    if not environment["matched"]:
        report["remaining"].insert(0, "CPU uv.lock does not match declared/installed dependencies")
    write_json(output / "verification.json", report)
    with (output / "report.md").open("x") as stream:
        stream.write(
            f"CPU checks passed: **{cpu_passed}**. Current-interpreter handoff checks: "
            f"**{report['cpu_handoff_ready']}**.\n\n"
            f"Clean frozen rebuild verified: **{report['clean_frozen_sync_verified']}**.\n\n"
            "Checks ran in the included source snapshot. See verification.json and logs. "
            "The ten-job fixture, ASL poses, costs and provider responses are synthetic. "
            "ASL inspection and two-arm launch preparation are recorded under demo/. "
            "Prepared commands were not executed. "
            "No runtime/scientific admission or rental authority follows.\n\n"
            + "\n".join(f"- {item}" for item in report["remaining"]) + "\n"
        )
    write_json(output / "checksums.json", receipts_for(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-only", action="store_true", help="fixture only; no suite/lock check")
    args = parser.parse_args(argv)
    try:
        result = run_demo(args.output) if args.demo_only else run_frozen_checks(args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "cloud_launch_authorized": False}))
        return 2
    print(json.dumps({key: result[key] for key in (
        ("passed", "jobs", "runtime_qualified") if args.demo_only else
        ("cpu_checks_passed", "cpu_handoff_ready", "runtime_qualified")
    )}))
    return 0 if result.get("passed", result.get("cpu_handoff_ready", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
