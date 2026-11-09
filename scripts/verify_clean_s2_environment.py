"""Install the unchanged CPU lock in a new environment and retain honest evidence.

Uses an explicitly identified uv executable, a new empty cache, and an existing
Python interpreter. No Python download, GPU action, or scientific admission.
Run with --output evidence/<new-name>; existing evidence is never overwritten.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREES = ("gate", "teacher_rl", "tests", "scripts", "experiments", "configs")
ENVIRONMENT_KEYS = (
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "COMSPEC", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def source_snapshot(root):
    paths = {root / "pyproject.toml", root / "uv.lock"}
    for name in (".python-version", ".claude/starred-cores.txt", "DOC_AUTHORITY.md"):
        if (root / name).exists():
            paths.add(root / name)
    for directory in SOURCE_TREES:
        tree = root / directory
        if tree.is_symlink():
            raise ValueError(f"Source directory is a symlink: {directory}")
        for path in tree.rglob("*"):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Source path is a symlink: {path.relative_to(root)}")
            if path.is_file():
                paths.add(path)
    paths.update((root / "docs").glob("S2*.md"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source snapshot refuses symlinks")
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(paths)}


def snapshot_hashes(snapshot):
    return {name: hashlib.sha256(value).hexdigest() for name, value in snapshot.items()}


def lock_candidates(lock):
    """Every lock entry per canonical package name (a name may have several, see below)."""
    candidates = {}
    for package in lock["package"]:
        candidates.setdefault(canonical(package["name"]), []).append(package)
    return candidates


def select_locked_package(candidates, name, edge, marker_matches):
    """Return the one lock entry an edge resolves to; any remaining ambiguity fails closed.

    uv writes one entry per source when ``[tool.uv.sources]`` splits a requirement by a
    marker (D-024: torch from PyPI off Linux and from the PyTorch CPU index on Linux). The
    edge then names ``version`` and ``source``, and each entry carries
    ``resolution-markers``. Entries are narrowed by the edge's version and source, then, if
    several remain, to those whose resolution markers match this environment; anything but
    exactly one entry raises.
    """
    options = list(candidates.get(name, ()))
    if not options:
        raise ValueError(f"Missing locked package: {name}")
    if "version" in edge:
        options = [package for package in options if package.get("version") == edge["version"]]
        if not options:
            raise ValueError(f"Inconsistent locked dependency: {name}")
    if "source" in edge:
        options = [package for package in options if package.get("source") == edge["source"]]
    if len(options) > 1:
        options = [package for package in options if package.get("resolution-markers")
                   and any(marker_matches(marker) for marker in package["resolution-markers"])]
    if len(options) != 1:
        raise ValueError(f"Unsupported ambiguous locked package: {name}")
    return options[0]


def source_splits(project):
    """``{canonical name: [(index URL, marker)]}`` for marker-split ``[tool.uv.sources]``."""
    tool = project.get("tool", {}).get("uv", {})
    urls = {entry["name"]: entry["url"] for entry in tool.get("index", [])}
    splits = {}
    for name, entries in tool.get("sources", {}).items():
        entries = [entries] if isinstance(entries, dict) else entries
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Unsupported uv source: {name}")
        rows = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"index", "marker"} \
                    or entry["index"] not in urls:
                raise ValueError(f"Unsupported uv source: {name}")
            rows.append((urls[entry["index"]], entry["marker"]))
        splits[canonical(name)] = rows
    return splits


def collapse_source_splits(items, splits, normalize_marker, marker_matches):
    """Map lock requirement metadata back to the declared requirements; fail closed.

    A marker-split source becomes one lock item per declared source (with ``index``) plus
    one complement item. Each split group must share one specifier, contain exactly the
    declared source items, exactly one complement, and exactly one item applicable here; it
    then stands for the single declared requirement ``{name, specifier}``. Items of names
    without a split are returned unchanged and may not carry ``index``.
    """
    plain, groups = [], {}
    for item in items:
        if not isinstance(item, dict) or set(item) - {"name", "specifier", "marker", "index"}:
            raise ValueError("Unsupported locked requirement metadata")
        name = canonical(item["name"])
        if name in splits:
            groups.setdefault(name, []).append(item)
        elif "index" in item:
            raise ValueError("Unsupported locked requirement metadata")
        else:
            plain.append(item)
    for name, group in sorted(groups.items()):
        specifiers = {item.get("specifier", "") for item in group}
        indexed = sorted((item["index"], normalize_marker(item.get("marker", "")))
                         for item in group if "index" in item)
        wanted = sorted((url, normalize_marker(marker)) for url, marker in splits[name])
        complements = [item for item in group if "index" not in item]
        if len(specifiers) != 1 or indexed != wanted or len(complements) != 1:
            raise ValueError(f"Inconsistent source split: {name}")
        if sum(bool(marker_matches(item.get("marker", ""))) for item in group) != 1:
            raise ValueError(f"Source split does not partition this platform: {name}")
        plain.append({"name": group[0]["name"], "specifier": specifiers.pop()})
    return plain


def locked_graph(lock, project_name, marker_matches):
    """Traverse all runtime/dev edges; unsupported ambiguous locks fail closed."""
    candidates = lock_candidates(lock)
    roots = candidates.get(canonical(project_name), [])
    if len(roots) != 1:
        raise ValueError(f"Unsupported ambiguous locked package: {canonical(project_name)}")
    root = roots[0]
    pending = list(root.get("dependencies", []))
    for group in root.get("dev-dependencies", {}).values():
        pending.extend(group)
    expected = {}
    while pending:
        edge = pending.pop()
        if "marker" in edge and not marker_matches(edge["marker"]):
            continue
        if edge.get("extra") or edge.get("extras"):
            raise ValueError("Optional-extra lock edges need explicit verifier support")
        name = canonical(edge["name"])
        package = select_locked_package(candidates, name, edge, marker_matches)
        if name in expected:
            if expected[name] != package["version"]:
                raise ValueError(f"Inconsistent locked dependency: {name}")
            continue
        expected[name] = package["version"]
        pending.extend(package.get("dependencies", []))
    return dict(sorted(expected.items()))


def compare_inventory(expected, installed):
    names = sorted(set(expected) | set(installed))
    return {
        "matched": expected == installed,
        "packages": {
            name: {"locked": expected.get(name), "installed": installed.get(name)} for name in names
        },
    }


def declaration_check(project, lock):
    """Verify frozen lock freshness without running a second resolver."""
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    issues = []
    try:
        declaration = project["project"]
        roots = [item for item in lock["package"]
                 if canonical(item["name"]) == canonical(declaration["name"])]
        if len(roots) != 1:
            raise ValueError("Missing or ambiguous project package")
        root = roots[0]
        if root.get("version") != declaration.get("version"):
            issues.append("project_version_differs_from_lock")
        if declaration.get("optional-dependencies") or declaration.get("dynamic"):
            raise ValueError("Optional/dynamic project requirements need explicit verifier support")
        if SpecifierSet(declaration["requires-python"]) != SpecifierSet(lock["requires-python"]):
            issues.append("python_constraint_differs_from_lock")

        def parsed(value):
            requirement = Requirement(value)
            if requirement.url or requirement.extras:
                raise ValueError("URL/extra requirements need explicit verifier support")
            return (
                canonical(requirement.name), str(requirement.specifier),
                str(requirement.marker) if requirement.marker else "",
            )

        splits = source_splits(project)

        def applies(marker):
            return Marker(marker).evaluate() if marker else True

        def normalized(marker):
            return str(Marker(marker)) if marker else ""

        def locked(items):
            items = collapse_source_splits(items, splits, normalized, applies)
            if any(set(item) - {"name", "specifier", "marker"} for item in items):
                raise ValueError("Unsupported locked requirement metadata")
            return sorted(parsed(
                item["name"] + item.get("specifier", "")
                + ("; " + item["marker"] if item.get("marker") else "")
            ) for item in items)

        candidates = lock_candidates(lock)

        def compare_group(label, requirements, metadata_items, resolved_edges):
            declared = sorted(parsed(value) for value in requirements)
            if declared != locked(metadata_items):
                issues.append(f"{label}_declarations_differ_from_lock")
            if {item[0] for item in declared} != {
                canonical(edge["name"]) for edge in resolved_edges
            }:
                issues.append(f"{label}_resolved_dependency_set_differs")
            for name, specifier, marker in declared:
                if marker and not Marker(marker).evaluate():
                    continue
                # Every lock entry of the name (one per split source) must satisfy it.
                versions = [package["version"] for package in candidates.get(name, ())]
                if not versions or any(v not in SpecifierSet(specifier) for v in versions):
                    issues.append(f"{label}_resolved_version_violates_requirement:{name}")

        metadata = root["metadata"]
        compare_group("runtime", declaration.get("dependencies", []),
                      metadata.get("requires-dist", []), root.get("dependencies", []))
        groups = project.get("dependency-groups", {})
        metadata_groups = metadata.get("requires-dev", {})
        resolved_groups = root.get("dev-dependencies", {})
        if set(groups) != set(metadata_groups) or set(groups) != set(resolved_groups):
            issues.append("dev_group_set_differs_from_lock")
        for name, requirements in groups.items():
            compare_group(f"dev:{name}", requirements, metadata_groups.get(name, []),
                          resolved_groups.get(name, []))
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"unsupported_or_incomplete_declarations:{exc}")
    return {"matched": not issues, "issues": issues}


def probe_environment(root, expected_environment):
    """Runs inside the newly installed interpreter, never the launching venv."""
    from packaging.markers import Marker
    from packaging.specifiers import SpecifierSet

    project_file = tomllib.loads((root / "pyproject.toml").read_text())
    project = project_file["project"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    declarations = declaration_check(project_file, lock)
    expected = locked_graph(lock, project["name"], lambda value: Marker(value).evaluate())
    installed, locations, duplicates = {}, {}, []
    for distribution in metadata.distributions():
        name = canonical(distribution.metadata["Name"])
        if name in installed:
            duplicates.append(name)
        installed[name] = distribution.version
        locations[name] = str(Path(distribution.locate_file("")).resolve())
    expected_environment = expected_environment.resolve()
    config = (expected_environment / "pyvenv.cfg").read_text().lower()
    isolated = (
        Path(sys.prefix).resolve() == expected_environment
        and sys.prefix != sys.base_prefix
        and site.ENABLE_USER_SITE is False
        and re.search(r"(?m)^include-system-site-packages\s*=\s*false\s*$", config) is not None
        and all(Path(path).is_relative_to(expected_environment) for path in locations.values())
    )
    report = compare_inventory(expected, installed)
    python_matches = platform.python_version() in SpecifierSet(project["requires-python"])
    report.update({
        "matched": report["matched"] and not duplicates and python_matches and declarations["matched"],
        "declarations": declarations,
        "isolated": isolated,
        "python": platform.python_version(),
        "python_matches_project": python_matches,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "executable": sys.executable,
        "package_locations": locations,
        "duplicates": duplicates,
        "user_site_enabled": site.ENABLE_USER_SITE,
    })
    return report


def run_command(name, command, *, cwd, environment, timeout, output):
    start = time.monotonic()
    timed_out = False
    launched = True
    try:
        result = subprocess.run(
            command, cwd=cwd, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")

        code, stdout = None, decode(exc.stdout)
        stderr = decode(exc.stderr) + f"\nCommand timed out after {timeout} seconds.\n"
        timed_out = True
    except OSError as exc:
        code, stdout, stderr = None, "", f"Unable to execute: {exc}\n"
        launched = False
    log = stdout + stderr
    with (output / f"{name}.log").open("x", encoding="utf-8") as stream:
        stream.write(log)
    return {
        "name": name, "command": command, "returncode": code, "timed_out": timed_out,
        "launched": launched,
        "timeout_seconds": timeout, "elapsed_seconds": round(time.monotonic() - start, 3),
    }, stdout if not timed_out else log


def verify(root, output, uv, python, install_timeout=300, check_timeout=900):
    root, output, uv, python = (Path(path).resolve() for path in (root, output, uv, python))
    if output.is_relative_to(root) and not output.is_relative_to(root / "evidence"):
        raise ValueError("Output cannot be inside a source/configuration directory")
    if not 0 < install_timeout <= 3600 or not 0 < check_timeout <= 3600:
        raise ValueError("Timeouts must be positive and at most 3600 seconds")
    before = source_snapshot(root)
    before_hashes = snapshot_hashes(before)
    uv_hash = sha256(uv)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, content in before.items():
            archive.writestr(name, content)
    temporary = Path(tempfile.mkdtemp(prefix="physloop-clean-s2-"))
    target, cache = temporary / "venv", temporary / "uv-cache"
    cache.mkdir()
    environment = {name: os.environ[name] for name in ENVIRONMENT_KEYS if name in os.environ}
    environment.update({
        "UV_CACHE_DIR": str(cache), "UV_PROJECT_ENVIRONMENT": str(target),
        "UV_PYTHON_DOWNLOADS": "never", "UV_NO_MANAGED_PYTHON": "1",
        "UV_HTTP_RETRIES": "0", "UV_HTTP_TIMEOUT": "10",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "UV_KEYRING_PROVIDER": "disabled",
    })
    records = []

    def run(name, command, timeout):
        record, stdout = run_command(
            name, command, cwd=root, environment=environment, timeout=timeout, output=output
        )
        records.append(record)
        return record["returncode"] == 0, stdout

    version_ok, version_text = run("uv-version", [str(uv), "--version"], 10)
    install_ok = False
    if version_ok:
        install_ok, _ = run("install", [
            str(uv), "sync", "--frozen", "--all-groups", "--no-install-project",
            "--no-managed-python", "--no-python-downloads", "--no-build", "--no-config",
            "--python", str(python), "--project", str(root),
        ], install_timeout)
    inventory, inventory_error = None, None
    fresh_tests_executed = False
    if install_ok:
        target_python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        inventory_ok, inventory_text = run("inventory", [
            str(target_python), "-I", "-B", str(Path(__file__).resolve()),
            "--probe", str(root), str(target),
        ], 30)
        try:
            inventory = json.loads(inventory_text) if inventory_ok else None
        except (TypeError, ValueError) as exc:
            inventory_error = str(exc)
        if (
            isinstance(inventory, dict) and inventory.get("matched") is True
            and inventory.get("isolated") is True
        ):
            run("tests", [str(target_python), "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"], check_timeout)
            fresh_tests_executed = records[-1].get(
                "launched", records[-1]["returncode"] is not None or records[-1]["timed_out"]
            )
            run("lint", [str(target_python), "-B", "-m", "ruff", "check", "--no-cache", "gate", "teacher_rl", "tests", "scripts"], check_timeout)
    source_error = None
    try:
        after_hashes = snapshot_hashes(source_snapshot(root))
        unchanged = before_hashes == after_hashes
        uv_unchanged = uv_hash == sha256(uv)
    except (OSError, ValueError) as exc:
        after_hashes, unchanged, uv_unchanged, source_error = None, False, False, str(exc)
    clean = bool(
        install_ok and isinstance(inventory, dict) and inventory.get("matched") is True
        and inventory.get("isolated") is True and unchanged and uv_unchanged
    )
    passed = clean and fresh_tests_executed and all(item["returncode"] == 0 for item in records)
    report = {
        "schema_version": 1, "scope": "S2 CPU environment installation and CPU source checks only",
        "created_at": datetime.now(UTC).isoformat(), "passed": passed,
        "clean_frozen_sync_verified": clean, "fresh_tests_executed": fresh_tests_executed,
        "source_sha256_before": before_hashes, "source_sha256_after": after_hashes,
        "sources_unchanged": unchanged, "source_recheck_error": source_error,
        "uv": {"path": str(uv), "sha256": uv_hash, "version_output": version_text.strip(), "unchanged": uv_unchanged},
        "python_requested": str(python), "python_downloads_allowed": False,
        "environment_directory": str(target),
        "cache_provenance": {"path": str(cache), "mode": "new_empty_cache", "empty_before_install": True, "copied_packages": False},
        "inventory": inventory, "inventory_error": inventory_error, "commands": records,
        "runtime_qualified": False, "cloud_launch_authorized": False,
        "limitations": [
            "CPU dependency installation does not qualify a Linux/CUDA simulator environment.",
            "Successful installation depends on the identified uv executable and configured registry integrity.",
            "No rented host, licensed assets, model load, simulator run or scientific admission.",
        ],
    }
    write_json(output / "verification.json", report)
    write_json(output / "checksums.json", {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
    })
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--probe"]:
        if len(argv) != 3:
            raise ValueError("Probe requires source root and expected environment")
        print(json.dumps(probe_environment(Path(argv[1]), Path(argv[2]))))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--install-timeout", type=int, default=300)
    parser.add_argument("--check-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    uv = Path(shutil.which(args.uv) or args.uv).resolve(strict=True)
    python = Path(shutil.which(args.python) or args.python).resolve(strict=True)
    report = verify(ROOT, args.output, uv, python, args.install_timeout, args.check_timeout)
    print(json.dumps({
        "passed": report["passed"], "evidence": str(args.output.resolve()),
        "clean_frozen_sync_verified": report["clean_frozen_sync_verified"],
        "fresh_tests_executed": report["fresh_tests_executed"],
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
