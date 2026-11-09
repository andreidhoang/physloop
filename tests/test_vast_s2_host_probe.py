"""Agent-authored known answers for the host probe (R2 H1-H3, R4 P1, profiler environment).

The probe runs under a sandboxed PATH: every host, container, GPU, network and toolchain command
is a fake that records its argv, and only benign file utilities are linked from the system. No
GPU, container engine, profiler, provider or network is touched. Assertions fail if any command
the probe uses is neither faked nor linked ("command not found" anywhere in the outputs).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/vast_s2_host_probe.sh"
IMAGE_ID = "sha256:" + "a" * 64
IMAGE_REF = "nvidia/cuda@sha256:" + "c" * 64
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
# Benign utilities the probe may use for real; everything else is faked below.
LINKED = ("bash", "sh", "cat", "cp", "date", "mkdir", "mktemp", "rm", "readlink", "grep", "env",
          "sha256sum", "head", "tr", "dirname", "basename", "sleep", "chmod", "ls")
RECORDED = ("nvidia-smi", "nsys", "ncu", "nvidia-ctk", "uv", "rustc", "git", "df", "nproc",
            "free", "ss", "curl", "systemctl", "iptables")
H1_CHECKS = ("nvidia_ctk_version", "uv_version", "rustc_version", "git_version",
             "docker_storage", "docker_root_free", "data_root_free", "cpu_count", "memory_bytes",
             "shm_size", "ulimits", "listening_sockets", "firewall_state", "compose_up_help",
             "hf_reachability", "nvcr_reachability")

DOCKER_FAKE = f"""#!/bin/sh
for arg in "$@"; do printf '%s\\t' "$arg"; done >> "$PHYSLOOP_TEST_SEEN/docker.log"
printf '\\n' >> "$PHYSLOOP_TEST_SEEN/docker.log"
eval last=\\${{$#}}
case "$1 $2" in
"image inspect")
    case "$last" in
    *"range .RepoDigests"*) printf '%s\\n%s\\n' "{IMAGE_ID}" "$3";;
    "{{{{.Id}}}}") printf '%s\\n' "{IMAGE_ID}";;
    *) printf '%s ["%s"] linux/amd64\\n' "{IMAGE_ID}" "$3";;
    esac;;
"info --format")
    case "$3" in
    "{{{{.DockerRootDir}}}}") echo /var/lib/docker;;
    *) echo "driver=overlay2 root=/var/lib/docker";;
    esac;;
"compose up") [ "$3" = --help ] && echo "      --pull string   Pull image before running";;
"version --format") echo "client=29.0.3 server=29.0.3";;
esac
exit 0
"""


def fake(folder, name, body):
    path = folder / name
    path.write_text(body if body.startswith("#!") else "#!/bin/sh\n" + body)
    path.chmod(0o755)


def recorder(seen, tool):
    return f'{{ printf "%s\\n" "ARGV $*"; env; }} > "{seen}/{tool}-$$.env"\nexit 0\n'


def sandbox(tmp_path):
    bin_dir, seen, home = tmp_path / "bin", tmp_path / "seen", tmp_path / "home"
    for folder in (bin_dir, seen, home):
        folder.mkdir()
    for tool in LINKED:
        real = shutil.which(tool, path="/usr/bin:/bin:/usr/sbin:/sbin")
        if real:
            (bin_dir / tool).symlink_to(real)
    if not (bin_dir / "sha256sum").exists():
        fake(bin_dir, "sha256sum", 'echo "0000  $1"\n')
    # Record every timeout invocation, then drop its options and duration and run at once.
    fake(bin_dir, "timeout", f'echo "ARGV $*" >> "{seen}/timeout.log"\n'
         'while [ $# -gt 0 ]; do case $1 in -k|-s) shift 2;; -*) shift;; '
         '*) break;; esac; done\nshift\nexec "$@"\n')
    fake(bin_dir, "uname", 'case "$1" in -s) echo Linux;; -m) echo x86_64;; '
         '*) echo "Linux 6.8.0 x86_64 GNU/Linux";; esac\n')
    fake(bin_dir, "docker", DOCKER_FAKE)
    for tool in RECORDED:
        fake(bin_dir, tool, recorder(seen, tool))
    return bin_dir, seen, home


def run_probe(tmp_path, uid=0, image_ref=IMAGE_REF, data_root=None, args=None):
    bin_dir, seen, home = sandbox(tmp_path)
    fake(bin_dir, "id", f'[ "$1" = -u ] && echo {uid} && exit 0\nexit 1\n')
    # Ubuntu's sudo PAM stack can add /etc/environment back after sudo resets the environment.
    fake(bin_dir, "sudo", f'[ "$1" = -n ] && shift\nexport {PAM_ADDED[0]}="{PAM_ADDED[1]}"\n'
         f'env > "{seen}/sudo-$$.env"\nexec "$@"\n')
    if data_root is None:
        data_root = tmp_path / "data-root"
        data_root.mkdir()
    out = tmp_path / "out"
    env = {"PATH": str(bin_dir), "HOME": str(home), "LANG": "C", "CUDA_VISIBLE_DEVICES": "0",
           "PHYSLOOP_TEST_SEEN": str(seen), **CREDENTIALS}
    argv = args if args is not None else [str(out), image_ref, str(data_root)]
    result = subprocess.run([str(bin_dir / "bash"), str(SCRIPT), *argv], env=env,
                            capture_output=True, text=True, timeout=120, check=False)
    return result, out, seen


def records(seen, tool):
    found = {}
    for path in sorted(seen.glob(f"{tool}-*.env")):
        lines = path.read_text().splitlines()
        found[lines[0]] = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    return found


def check_rows(out):
    return dict(line.split("\t")[0::2] for line in (out / "checks.tsv").read_text().splitlines())


def docker_calls(seen):
    return [line.rstrip("\t").split("\t")
            for line in (seen / "docker.log").read_text().splitlines()]


@pytest.fixture(scope="module")
def root_run(tmp_path_factory):
    """One full root-user probe run shared by the read-only assertions below (each run ~8 s)."""
    tmp_path = tmp_path_factory.mktemp("probe-root")
    result, out, seen = run_probe(tmp_path, 0)
    return tmp_path, result, out, seen


@pytest.fixture(scope="module")
def sudo_run(tmp_path_factory):
    """A non-root run whose data root does not exist yet."""
    tmp_path = tmp_path_factory.mktemp("probe-sudo")
    result, out, seen = run_probe(tmp_path, 1000, data_root=tmp_path / "absent")
    return tmp_path, result, out, seen


def assert_no_missing_tools(result, out):
    texts = [result.stderr, result.stdout]
    texts += [path.read_text(errors="replace") for path in out.glob("*.log")]
    for text in texts:
        assert "command not found" not in text and "No such file or directory: " not in text


@pytest.mark.parametrize("uid", [0, 1000], ids=["root", "sudo"])
def test_profilers_get_only_allowlisted_environment(root_run, sudo_run, uid):
    tmp_path, result, out, seen = root_run if uid == 0 else sudo_run
    # Required OS identity checks fail off-Linux (no /etc/os-release); profiler rows matter here.
    assert result.returncode in (0, 2), result.stderr
    rows = check_rows(out)
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
    assert_no_missing_tools(result, out)


def test_public_image_is_pulled_only_by_digest(root_run):
    _, result, out, seen = root_run
    calls = docker_calls(seen)
    pulls = [call for call in calls if call[:1] == ["pull"]]
    assert pulls == [["pull", IMAGE_REF]]
    for call in calls:
        for arg in call:
            if "nvidia/cuda" in arg:
                assert arg == IMAGE_REF, call
    assert (out / "Dockerfile").read_text().splitlines()[0] == f"FROM {IMAGE_REF}"
    manifest = (out / "manifest.txt").read_text()
    assert f"public_image_ref={IMAGE_REF}\n" in manifest
    assert f"resolved_image_id={IMAGE_ID}\n" in manifest
    assert "public_image_tag" not in manifest
    # Containers run the resolved local image ID, never the reference.
    runs = [call for call in calls if call[:1] == ["run"]]
    assert runs and all(IMAGE_ID in call for call in runs)
    assert_no_missing_tools(result, out)


@pytest.mark.parametrize("args", [
    ["OUT", "nvidia/cuda:12.8.1-devel-ubuntu22.04", "/data"],
    ["OUT", "nvidia/cuda@sha256:" + "C" * 64, "/data"],
    ["OUT", "sha256:" + "a" * 64, "/data"],
    ["OUT", IMAGE_REF, "relative/data"],
    ["OUT", IMAGE_REF],
    ["OUT"],
    ["relative/out", IMAGE_REF, "/data"],
])
def test_tag_or_malformed_arguments_exit_64_before_any_tool(tmp_path, args):
    out = tmp_path / "out"
    argv = [str(out) if a == "OUT" else a for a in args]
    result, _, seen = run_probe(tmp_path, args=argv)
    assert result.returncode == 64, result.stderr
    assert not out.exists()
    assert not (seen / "docker.log").exists() and not (seen / "timeout.log").exists()


def test_outer_kill_after_covers_longest_check_and_cleanup(root_run):
    _, _, out, seen = root_run
    outer = [line for line in (seen / "timeout.log").read_text().splitlines()
             if " 20m bash " in line]
    assert len(outer) == 1
    kill_after = int(re.search(r"--kill-after=(\d+)s", outer[0]).group(1))
    assert "--signal=TERM" in outer[0]

    # Independent oracle from the script text: every foreground check's timeout and kill-after,
    # and the sum of every bounded command inside cleanup().
    text = SCRIPT.read_text()
    checks = [int(n) for n in re.findall(r"^\s*(?:if !? ?)?check (?:required|diagnostic) \S+ "
                                         r"(\d+) ", text, flags=re.MULTILINE)]
    check_kill = int(re.search(r'timeout --signal=TERM --kill-after=(\d+)s "\$\{seconds\}s"',
                               text).group(1))
    body = text[text.index("cleanup() {"):text.index("trap cleanup EXIT")]
    cleanup = sum(int(k) + int(t) for k, t in re.findall(r"timeout -k (\d+)s (\d+)s", body))
    assert max(checks) == 600 and check_kill == 5 and cleanup == 54
    assert kill_after > max(checks) + check_kill + cleanup
    assert f"outer_kill_after_s={kill_after}\n" in (out / "manifest.txt").read_text()


def test_h1_inventory_records_and_h3_host_network_compose(root_run):
    tmp_path, result, out, seen = root_run
    rows = check_rows(out)
    assert {name: rows.get(name) for name in H1_CHECKS} == dict.fromkeys(H1_CHECKS, "0")
    for name in ("compose_host_config", "compose_host_network_device_ids"):
        assert rows[name] == "0"
    manifest = (out / "manifest.txt").read_text()
    assert "compose_up_pull_flag=listed\n" in manifest
    assert f"data_root={tmp_path / 'data-root'}\n" in manifest
    urls = {line.split()[-1] for view in records(seen, "curl") for line in [view]}
    assert urls == {"https://huggingface.co/api/models/nvidia/Alpamayo-1.5-10B",
                    "https://nvcr.io/v2/"}
    df_targets = {argv.split()[-1] for argv in records(seen, "df")}
    assert {"/var/lib/docker", str(tmp_path / "data-root"), "/dev/shm"} <= df_targets
    compose_host = (out / "compose-host.yaml").read_text()
    assert "network_mode: host" in compose_host and "pid: host" in compose_host
    assert 'device_ids: ["0"]' in compose_host and "pull_policy: never" in compose_host
    calls = docker_calls(seen)
    project = next(c[2] for c in calls if c[:2] == ["compose", "-p"] and c[2].endswith("-host"))
    downs = [c for c in calls if c[:3] == ["compose", "-p", project] and "down" in c]
    assert len(downs) == 1
    assert_no_missing_tools(result, out)


def test_missing_data_root_is_recorded_unverified_not_passed(sudo_run):
    _, result, out, _ = sudo_run
    assert "data_root_free" not in check_rows(out)
    assert "UNVERIFIED" in (out / "data-root-unverified.log").read_text()
    assert_no_missing_tools(result, out)
