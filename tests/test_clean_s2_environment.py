"""Negative controls for the fresh-environment verifier, using fake tool results."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_clean_s2_environment.py"
    spec = importlib.util.spec_from_file_location("clean_s2_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "0.0.0"\nrequires-python = "==3.12.*"\n'
    )
    (root / "uv.lock").write_text('version = 1\n')
    (root / "gate").mkdir()
    (root / "gate/example.py").write_text("value = 1\n")
    return root


def fake_tools(module, monkeypatch, *, install_ok=True, probe_ok=True, mutate=None):
    calls = []

    def run(name, command, *, cwd, environment, timeout, output):
        calls.append((name, command, environment, timeout))
        stdout = "uv test fixture\n" if name == "uv-version" else ""
        code = 0
        if name == "install":
            code = 0 if install_ok else 101
            stdout = "" if install_ok else "SystemConfiguration NULL fixture\n"
            if mutate is not None:
                mutate()
        if name == "inventory":
            stdout = json.dumps(
                {"matched": probe_ok, "isolated": probe_ok, "packages": {"example": "1"}}
            )
        (output / f"{name}.log").write_text(stdout)
        return {"name": name, "command": command, "returncode": code, "timed_out": False}, stdout

    monkeypatch.setattr(module, "run_command", run)
    return calls


def test_failed_install_never_runs_probe_or_tests(project, tmp_path, monkeypatch):
    module = verifier()
    calls = fake_tools(module, monkeypatch, install_ok=False)
    report = module.verify(project, project / "evidence/failed", Path(__file__), Path(__file__))
    assert not report["clean_frozen_sync_verified"]
    assert not report["fresh_tests_executed"]
    assert not report["passed"]
    assert [item[0] for item in calls] == ["uv-version", "install"]
    assert (project / "evidence/failed/checksums.json").is_file()


def test_successful_exit_with_wrong_inventory_blocks_checks(project, monkeypatch):
    module = verifier()
    calls = fake_tools(module, monkeypatch, probe_ok=False)
    report = module.verify(project, project / "evidence/wrong", Path(__file__), Path(__file__))
    assert not report["passed"]
    assert not report["clean_frozen_sync_verified"]
    assert not report["fresh_tests_executed"]
    assert [item[0] for item in calls] == ["uv-version", "install", "inventory"]


def test_positive_fixture_uses_only_new_interpreter_and_empty_cache(project, monkeypatch):
    module = verifier()
    calls = fake_tools(module, monkeypatch)
    report = module.verify(project, project / "evidence/ok", Path(__file__), Path(__file__))
    assert report["passed"]
    assert report["clean_frozen_sync_verified"]
    assert report["fresh_tests_executed"]
    install = next(item for item in calls if item[0] == "install")
    assert {"--frozen", "--no-managed-python", "--no-python-downloads", "--no-install-project"} <= set(install[1])
    assert install[2]["UV_PYTHON_DOWNLOADS"] == "never"
    assert "PYTHONPATH" not in install[2]
    assert report["cache_provenance"]["empty_before_install"]
    target = str(Path(report["environment_directory"]) / "bin/python")
    for name, command, _, timeout in calls:
        assert timeout > 0
        if name in ("inventory", "tests", "lint"):
            assert command[0] == target
            assert command[0] != str(Path(__file__))
    assert "-I" in next(item[1] for item in calls if item[0] == "inventory")


def test_changed_source_is_rejected(project, monkeypatch):
    module = verifier()
    fake_tools(
        module,
        monkeypatch,
        mutate=lambda: (project / "gate/example.py").write_text("value = 2\n"),
    )
    report = module.verify(project, project / "evidence/changed", Path(__file__), Path(__file__))
    assert not report["passed"]
    assert not report["sources_unchanged"]
    assert not report["clean_frozen_sync_verified"]


def test_existing_output_and_source_destinations_are_refused(project, monkeypatch):
    module = verifier()
    fake_tools(module, monkeypatch)
    existing = project / "evidence/existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.verify(project, existing, Path(__file__), Path(__file__))
    with pytest.raises(ValueError, match="source"):
        module.verify(project, project / "gate/new", Path(__file__), Path(__file__))


def test_timeout_is_retained_as_failure(tmp_path, monkeypatch):
    module = verifier()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"partial\n")

    monkeypatch.setattr(module.subprocess, "run", timeout)
    result, log = module.run_command(
        "timeout", ["fake"], cwd=tmp_path, environment={}, timeout=1, output=tmp_path
    )
    assert result["returncode"] is None
    assert result["timed_out"]
    assert "partial" in log
    assert (tmp_path / "timeout.log").read_text() == log


def test_inventory_rejects_transitive_mismatch_and_unexpected_package():
    module = verifier()
    lock = {
        "package": [
            {"name": "example", "dependencies": [{"name": "direct"}]},
            {"name": "direct", "version": "1", "dependencies": [{"name": "nested"}]},
            {"name": "nested", "version": "2"},
        ]
    }
    expected = module.locked_graph(lock, "example", lambda marker: True)
    assert expected == {"direct": "1", "nested": "2"}
    assert not module.compare_inventory(expected, {"direct": "1", "nested": "1"})["matched"]
    assert not module.compare_inventory(expected, {**expected, "unexpected": "3"})["matched"]
    assert module.compare_inventory(expected, expected)["matched"]


def test_lock_markers_and_ambiguous_versions_fail_closed():
    module = verifier()
    lock = {
        "package": [
            {"name": "example", "dependencies": [{"name": "platform", "marker": "skip"}]},
            {"name": "platform", "version": "1"},
        ]
    }
    assert module.locked_graph(lock, "example", lambda marker: marker != "skip") == {}
    lock["package"].append({"name": "platform", "version": "2"})
    with pytest.raises(ValueError, match="ambiguous"):
        module.locked_graph(lock, "example", lambda marker: True)


def test_truthy_non_boolean_inventory_is_rejected(project, monkeypatch):
    module = verifier()
    calls = fake_tools(module, monkeypatch, probe_ok="false")
    report = module.verify(project, project / "evidence/nonboolean", Path(__file__), Path(__file__))
    assert not report["clean_frozen_sync_verified"]
    assert not report["fresh_tests_executed"]
    assert all(item[0] != "tests" for item in calls)


def test_probe_rejects_system_site_packages(project, monkeypatch, tmp_path):
    module = verifier()
    (project / "uv.lock").write_text(
        'requires-python = "==3.12.*"\n[[package]]\nname = "example"\nversion = "0.0.0"\n'
        '[package.metadata]\nrequires-dist = []\n'
    )
    env = tmp_path / "target"
    env.mkdir()
    (env / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    monkeypatch.setattr(module.sys, "prefix", str(env))
    monkeypatch.setattr(module.sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(module.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(module.metadata, "distributions", list)
    report = module.probe_environment(project, env)
    assert report["matched"]
    assert not report["isolated"]


def test_source_symlink_is_rejected_before_installer(project, monkeypatch):
    module = verifier()
    calls = fake_tools(module, monkeypatch)
    (project / "gate/link.py").symlink_to(project / "gate/example.py")
    with pytest.raises(ValueError, match="symlink"):
        module.verify(project, project / "evidence/symlink", Path(__file__), Path(__file__))
    assert not calls


def test_unavailable_executable_is_not_counted_as_executed(tmp_path):
    module = verifier()
    result, _ = module.run_command(
        "missing", [str(tmp_path / "does-not-exist")], cwd=tmp_path,
        environment={}, timeout=1, output=tmp_path,
    )
    assert result["returncode"] is None
    assert not result["launched"]


@pytest.mark.parametrize("change", ["runtime", "group", "python", "resolved_version"])
def test_stale_project_cannot_pass_with_matching_installed_lock(change):
    module = verifier()
    project = {
        "project": {"name": "example", "version": "0.0.0", "requires-python": "==3.12.*",
                    "dependencies": ["direct==1"]},
        "dependency-groups": {"dev": ["nested==2"]},
    }
    lock = {
        "requires-python": "==3.12.*",
        "package": [
            {"name": "example", "version": "0.0.0", "dependencies": [{"name": "direct"}],
             "dev-dependencies": {"dev": [{"name": "nested"}]},
             "metadata": {"requires-dist": [{"name": "direct", "specifier": "==1"}],
                          "requires-dev": {"dev": [{"name": "nested", "specifier": "==2"}]}}},
            {"name": "direct", "version": "1"},
            {"name": "nested", "version": "2"},
        ],
    }
    assert module.declaration_check(project, lock)["matched"]
    if change == "runtime":
        project["project"]["dependencies"] = ["direct==9"]
    elif change == "group":
        project["dependency-groups"]["dev"] = ["nested==9"]
    elif change == "python":
        project["project"]["requires-python"] = ">=3.12"
    else:
        lock["package"][1]["version"] = "9"
    expected = module.locked_graph(lock, "example", lambda marker: True)
    assert module.compare_inventory(expected, expected)["matched"]
    assert not module.declaration_check(project, lock)["matched"]


def _split_lock():
    """A lock shaped like uv's output for a marker-split [tool.uv.sources] requirement."""
    pypi = {"registry": "https://pypi.org/simple"}
    cpu = {"registry": "https://download.pytorch.org/whl/cpu"}
    return {
        "package": [
            {"name": "example", "version": "0", "dev-dependencies": {"s4": [
                {"name": "torch", "version": "2.8.0", "source": pypi,
                 "marker": "sys_platform != 'linux'"},
                {"name": "torch", "version": "2.8.0+cpu", "source": cpu,
                 "marker": "sys_platform == 'linux'"},
            ]}},
            {"name": "torch", "version": "2.8.0", "source": pypi,
             "resolution-markers": ["sys_platform != 'linux'"]},
            {"name": "torch", "version": "2.8.0+cpu", "source": cpu,
             "resolution-markers": ["sys_platform == 'linux'"]},
        ]
    }


def test_marker_split_lock_selects_the_platform_entry():
    module = verifier()
    lock = _split_lock()
    mac = lambda marker: marker == "sys_platform != 'linux'"
    linux = lambda marker: marker == "sys_platform == 'linux'"
    assert module.locked_graph(lock, "example", mac) == {"torch": "2.8.0"}
    assert module.locked_graph(lock, "example", linux) == {"torch": "2.8.0+cpu"}


def test_marker_split_lock_without_distinguishing_edge_fails_closed():
    module = verifier()
    lock = _split_lock()
    for edge in lock["package"][0]["dev-dependencies"]["s4"]:
        edge.pop("version"), edge.pop("source")
    both = lambda marker: True  # both resolution markers match: ambiguous
    with pytest.raises(ValueError, match="ambiguous"):
        module.locked_graph(lock, "example", both)


def test_source_split_declarations_collapse_and_fail_closed():
    module = verifier()
    splits = {"torch": [("https://download.pytorch.org/whl/cpu", "sys_platform == 'linux'")]}
    same = lambda marker: marker
    mac = lambda marker: marker in ("", "sys_platform != 'linux'")
    items = [
        {"name": "torch", "specifier": "==2.8.0", "marker": "sys_platform != 'linux'"},
        {"name": "torch", "specifier": "==2.8.0", "marker": "sys_platform == 'linux'",
         "index": "https://download.pytorch.org/whl/cpu"},
        {"name": "safetensors", "specifier": "==0.7.0"},
    ]
    collapsed = module.collapse_source_splits(items, splits, same, mac)
    assert sorted(item["name"] for item in collapsed) == ["safetensors", "torch"]
    assert {"name": "torch", "specifier": "==2.8.0"} in collapsed
    with pytest.raises(ValueError, match="Inconsistent source split"):
        module.collapse_source_splits(items[:1] + items[2:], splits, same, mac)  # index item gone
    with pytest.raises(ValueError, match="partition"):
        module.collapse_source_splits(items, splits, same, lambda marker: True)
    with pytest.raises(ValueError, match="Unsupported locked requirement metadata"):
        module.collapse_source_splits([{"name": "numpy", "index": "x"}], splits, same, mac)


def test_repository_lock_declarations_match():
    """The real pyproject/uv.lock (s4 torch split by platform) pass the declaration check."""
    import tomllib

    module = verifier()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert module.declaration_check(project, lock) == {"matched": True, "issues": []}
