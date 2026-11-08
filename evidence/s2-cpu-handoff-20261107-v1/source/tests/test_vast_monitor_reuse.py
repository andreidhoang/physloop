"""CPU fake-command tests for portable, non-disruptive monitoring probes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
CUDA_REF = "registry.example.test/cuda@sha256:" + "a" * 64
CUDA_ID = "sha256:" + "b" * 64
EXPORTER_REF = (
    "nvcr.io/nvidia/k8s/dcgm-exporter@sha256:"
    "804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b"
)
EXPORTER_ID = "sha256:" + "c" * 64
PROBES = ("vast_monitor_control.sh", "vast_exporter_control.sh")

FAKE = r'''
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["SEEN"], "a") as stream:
    stream.write(json.dumps([name, *args]) + "\n")
if name == "docker":
    state_path = Path(os.environ["STATE"])
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args[:2] == ["image", "inspect"]:
        ref = args[2]
        if os.environ.get("INSPECT_FAIL") == "1":
            sys.exit(1)
        image = os.environ["CUDA_ID"] if ref == os.environ["CUDA_REF"] else os.environ["EXPORTER_ID"]
        print("invalid-id" if os.environ.get("INVALID_ID") == "1" else image)
        print("wrong@sha256:" + "d" * 64 if os.environ.get("WRONG_DIGEST") == "1" else ref)
    elif args and args[0] == "run":
        if os.environ.get("LATE_COLLISION") == "1":
            sys.exit(1)
        container_name = args[args.index("--name") + 1]
        cid = ("e" if "--cap-add" in args else "f") * 64
        label = args[args.index("--label") + 1] if "--label" in args else "missing"
        state[cid] = {"name": container_name, "label": label.split("=", 1)[-1]}
        state_path.write_text(json.dumps(state))
        if "--cidfile" in args:
            Path(args[args.index("--cidfile") + 1]).write_text(cid)
        if os.environ.get("RUN_FAIL") == "1":
            sys.exit(1)
        print(cid)
    elif args[:2] == ["container", "ls"]:
        if os.environ.get("INVENTORY_FAIL") == "1":
            sys.exit(1)
        constraint = args[args.index("--filter") + 1]
        if constraint.startswith("name="):
            if os.environ.get("COLLISION") == "1":
                print("a" * 64)
        else:
            cid = constraint.removeprefix("id=")
            if cid in state:
                print(cid)
    elif args[:2] == ["container", "inspect"]:
        cid = args[2]
        if cid not in state:
            sys.exit(1)
        print(cid)
        print("/" + state[cid]["name"])
        print("foreign-owner" if os.environ.get("WRONG_OWNER") == "1" else state[cid]["label"])
    elif args[:2] == ["rm", "-f"]:
        state.pop(args[2], None)
        state_path.write_text(json.dumps(state))
elif name == "systemctl":
    if args[:2] == ["is-active", "--quiet"]:
        sys.exit(int(os.environ.get("SERVICE_STATUS", "3")))
    sys.exit(91)  # Service mutations are never allowed by this fixture.
elif name == "pgrep":
    sys.exit(int(os.environ.get("PGREP_STATUS", "1")))
elif name == "curl":
    print("DCGM_FI_DEV_GPU_UTIL 80")
sys.exit(0)
'''


def run_probe(tmp_path, script, image_ref=CUDA_REF, **options):
    bin_dir, work = tmp_path / "bin", tmp_path / "work with spaces"
    bin_dir.mkdir()
    work.mkdir()
    (work / "tensor_control").write_text("#!/bin/sh\nexit 0\n")
    (work / "tensor_control").chmod(0o755)
    (work / "vast_tensor_control.cu").write_text("// fake source\n")
    timeout = bin_dir / "timeout"
    timeout.write_text(
        '#!/bin/sh\nwhile [ $# -gt 0 ]; do case $1 in -k|-s) shift 2;; '
        '-*) shift;; *) break;; esac; done\nshift\nexec "$@"\n'
    )
    timeout.chmod(0o755)
    for name in ("docker", "dcgmi", "nvidia-smi", "systemctl", "pgrep", "curl", "ss", "sleep"):
        fake = bin_dir / name
        fake.write_text(f"#!{sys.executable}\n" + FAKE)
        fake.chmod(0o755)
    seen, out = tmp_path / "seen.jsonl", tmp_path / "out"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SEEN": str(seen),
        "STATE": str(tmp_path / "containers.json"),
        "CUDA_REF": CUDA_REF,
        "CUDA_ID": CUDA_ID,
        "EXPORTER_ID": EXPORTER_ID,
        **options,
    }
    command = ["bash", str(SCRIPTS / script), str(out), str(work)]
    if image_ref is not None:
        command.append(image_ref)
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30,
                            check=False)
    calls = [json.loads(line) for line in seen.read_text().splitlines()] if seen.exists() else []
    return result, out, work, calls


@pytest.mark.parametrize("script", PROBES)
@pytest.mark.parametrize("image_ref", [None, "cuda:latest", "sha256:" + "a" * 64])
def test_missing_or_nonpullable_pin_rejected_before_host_commands(tmp_path, script, image_ref):
    result, out, _, calls = run_probe(tmp_path, script, image_ref)
    assert result.returncode == 64
    assert not out.exists()
    assert calls == []


@pytest.mark.parametrize("script", PROBES)
def test_cached_digest_resolves_to_local_id_without_pull(tmp_path, script):
    result, out, work, calls = run_probe(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert (out / "result.txt").exists()
    assert not any(call[:2] == ["docker", "pull"] for call in calls)
    runs = [call for call in calls if call[:2] == ["docker", "run"]]
    assert runs and any(CUDA_ID in call for call in runs)
    assert all("--pull=never" in call for call in runs)
    assert all(CUDA_REF not in call and EXPORTER_REF not in call for call in runs)
    assert any(f"type=bind,src={work},dst=/probe" in call for call in runs)
    manifest = (out / "manifest.txt").read_text()
    assert CUDA_REF in manifest and CUDA_ID in manifest
    if script == "vast_exporter_control.sh":
        assert any(EXPORTER_ID in call for call in runs)
    assert not any(call[0] == "systemctl" and call[1] in ("stop", "start") for call in calls)


@pytest.mark.parametrize("script", PROBES)
@pytest.mark.parametrize("failure", ["WRONG_DIGEST", "INVALID_ID", "INSPECT_FAIL"])
def test_unverified_local_image_never_runs(tmp_path, script, failure):
    result, out, _, calls = run_probe(tmp_path, script, **{failure: "1"})
    assert result.returncode == 2
    assert not (out / "result.txt").exists()
    assert not any(call[:2] == ["docker", "run"] for call in calls)


@pytest.mark.parametrize("options", [
    {"SERVICE_STATUS": "0"}, {"SERVICE_STATUS": "1"}, {"PGREP_STATUS": "0"},
    {"PGREP_STATUS": "2"},
])
def test_exporter_refuses_host_engine_or_unknown_service_state(tmp_path, options):
    result, out, _, calls = run_probe(tmp_path, "vast_exporter_control.sh", **options)
    assert result.returncode == 2
    assert not (out / "result.txt").exists()
    assert not any(call[:2] == ["docker", "run"] for call in calls)
    assert not any(call[0] == "systemctl" and call[1] in ("stop", "start") for call in calls)


@pytest.mark.parametrize("script", PROBES)
def test_workload_failure_runs_only_scoped_cleanup(tmp_path, script):
    result, out, _, calls = run_probe(tmp_path, script, RUN_FAIL="1")
    assert result.returncode == 2
    assert not (out / "result.txt").exists()
    cleanup = [call for call in calls if call[:3] == ["docker", "rm", "-f"]]
    assert cleanup
    assert all(len(cid) == 64 and set(cid) <= {"e", "f"} for call in cleanup for cid in call[3:])
    for call in cleanup:
        assert any(view[:3] == ["docker", "container", "inspect"] and view[3] == call[3]
                   for view in calls)
    assert not any(call[:2] == ["docker", "system"] for call in calls)


@pytest.mark.parametrize("script", PROBES)
@pytest.mark.parametrize("failure", ["COLLISION", "LATE_COLLISION", "INVENTORY_FAIL"])
def test_existing_or_uncertain_container_is_never_removed(tmp_path, script, failure):
    result, _, _, calls = run_probe(tmp_path, script, **{failure: "1"})
    assert result.returncode == 2
    assert not any(call[:2] == ["docker", "rm"] for call in calls)
    if failure != "LATE_COLLISION":
        assert not any(call[:2] == ["docker", "run"] for call in calls)


@pytest.mark.parametrize("script", PROBES)
def test_cleanup_requires_matching_ownership_label(tmp_path, script):
    result, out, _, calls = run_probe(tmp_path, script, WRONG_OWNER="1")
    assert result.returncode == 2
    assert not any(call[:2] == ["docker", "rm"] for call in calls)
    assert "cleanup_exit=2" in (out / "manifest.txt").read_text()
