"""Agent-authored known answers for the host probe's profiler environment; every host command is
a fake, so no GPU, container engine, profiler or provider is touched."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/vast_s2_host_probe.sh"
IMAGE_ID = "sha256:" + "a" * 64
# Shapes of what nsys stored from a real Vast login session; the values are fakes.
CREDENTIALS = {
    "CONTAINER_API_KEY": "fake-container-api-key",
    "JUPYTER_TOKEN": "fake-jupyter-token",
    "OPEN_BUTTON_TOKEN": "fake-open-button-token",
    "SSH_CLIENT": "203.0.113.7 50000 22",
    "SSH_CONNECTION": "203.0.113.7 50000 198.51.100.9 22",
    "PUBLIC_KEY": "ssh-ed25519 AAAAfake user@example.test",
    "SSH_PUBLIC_KEY": "ssh-ed25519 AAAAfake user@example.test",
}
PAM_ADDED = ("PAM_ADDED_SECRET", "fake-pam-environment-value")
KEPT = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES"}
SHELL_BOOKKEEPING = {"PWD", "OLDPWD", "SHLVL", "_"}
PROFILER_CHECKS = ("nsys_version", "nsys_environment", "nsys_cuda_trace", "ncu_version",
                   "ncu_actual_kernel")


def fake(folder, name, body):
    path = folder / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def recorder(seen, tool):
    return f'{{ echo "ARGV $*"; env; }} > "{seen}/{tool}-$$.env"\nexit 0\n'


def run_probe(tmp_path, uid):
    bin_dir, seen, home = tmp_path / "bin", tmp_path / "seen", tmp_path / "home"
    for folder in (bin_dir, seen, home):
        folder.mkdir()
    # Drop timeout's options and duration, then run the command at once.
    fake(bin_dir, "timeout", 'while [ $# -gt 0 ]; do case $1 in -k|-s) shift 2;; -*) shift;; '
         '*) break;; esac; done\nshift\nexec "$@"\n')
    fake(bin_dir, "id", f'[ "$1" = -u ] && echo {uid} && exit 0\nexit 1\n')
    # Ubuntu's sudo PAM stack can add /etc/environment back after sudo resets the environment.
    fake(bin_dir, "sudo", f'[ "$1" = -n ] && shift\nexport {PAM_ADDED[0]}="{PAM_ADDED[1]}"\n'
         f'env > "{seen}/sudo-$$.env"\nexec "$@"\n')
    fake(bin_dir, "docker", f'case "$*" in *"image inspect"*) echo "{IMAGE_ID}";; esac\nexit 0\n')
    for tool in ("nvidia-smi", "nsys", "ncu"):
        fake(bin_dir, tool, recorder(seen, tool))
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(home), "LANG": "C",
           "CUDA_VISIBLE_DEVICES": "0", **CREDENTIALS}
    out = tmp_path / "out"
    result = subprocess.run(["bash", str(SCRIPT), str(out)], env=env, capture_output=True,
                            text=True, timeout=60, check=False)
    return result, out, seen


def records(seen, tool):
    found = {}
    for path in sorted(seen.glob(f"{tool}-*.env")):
        lines = path.read_text().splitlines()
        found[lines[0]] = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    return found


@pytest.mark.parametrize("uid", [0, 1000], ids=["root", "sudo"])
def test_profilers_get_only_allowlisted_environment(tmp_path, uid):
    result, out, seen = run_probe(tmp_path, uid)
    # Required Linux and GPU checks fail off-Linux; only the profiler rows matter here.
    assert result.returncode in (0, 2), result.stderr
    rows = dict(line.split("\t")[0::2] for line in (out / "checks.tsv").read_text().splitlines())
    assert {name: rows.get(name) for name in PROFILER_CHECKS} == dict.fromkeys(PROFILER_CHECKS, "0")

    # Control: the credentials did reach the script, so their absence below is the fix.
    smi = records(seen, "nvidia-smi")
    assert smi and all(view["CONTAINER_API_KEY"] == CREDENTIALS["CONTAINER_API_KEY"]
                       for view in smi.values())
    sudo = [path.read_text() for path in seen.glob("sudo-*.env")]
    if uid == 0:
        assert not sudo
    else:
        assert any(f"{PAM_ADDED[0]}={PAM_ADDED[1]}" in text for text in sudo)

    profiled = {**records(seen, "nsys"), **records(seen, "ncu")}
    assert {argv.split()[1] for argv in profiled} >= {"--version", "status", "profile", "--set"}
    for argv, view in profiled.items():
        assert set(view) <= KEPT | SHELL_BOOKKEEPING, argv
        assert view["CUDA_VISIBLE_DEVICES"] == "0", argv
        assert view["PATH"].startswith(str(tmp_path / "bin")), argv
        text = "\n".join(view.values())
        for value in (*CREDENTIALS.values(), PAM_ADDED[1]):
            assert value not in text, argv
