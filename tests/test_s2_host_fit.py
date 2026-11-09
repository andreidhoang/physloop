"""Agent-authored known answers for the delivered-host fit check (F3; CPU-09, R2 A1/H1).

Probe texts are hand-written in the exact shapes the 2026-11-01 Vast host produced
(``evidence/s2-vast-host-20261101-v1/probe-v2/gpu_identity.log``, ``disk_space.log``) and the
GNU coreutils ``nproc`` / ``free -b`` / ``df -B1`` formats; every expected number is derived by
hand in a comment. No tool, provider or network is used.
"""

from copy import deepcopy

import pytest

from gate.s2_host_fit import (
    PROBE_FILES,
    REQUIREMENT_FIELDS,
    assess_host_fit,
    host_fit_report,
    parse_probe_facts,
    validate_requirements,
)
from gate.s2_identity import identity_hash

WS = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
GPU_LOG = (
    "index, name, uuid, driver_version, memory.total [MiB], pci.bus_id\n"
    f"0, {WS}, GPU-e7664bde-fdf4-a368-400a-7f1d6a3d9f57, 580.95.05, 97887 MiB, "
    "00000000:00:07.0\n")
# 97,887 MiB x 1,048,576 = 97,887,000,000 + 97,887 x 48,576 (= 4,754,958,912)
VRAM_BYTES = 102_641_958_912
CAP_LOG = "index, compute_cap\n0, 12.0\n"
FREE_LOG = (
    "               total        used        free      shared  buff/cache   available\n"
    "Mem:     104152227840  3209297920 97865084928     1196032  3077844992 100942929920\n"
    "Swap:               0           0           0\n")
DOCKER_DF = ("Filesystem        1B-blocks        Used    Available Use% Mounted on\n"
             "/dev/root      166013304832 27918352384 138078175232  17% /\n")
DATA_DF = ("Filesystem        1B-blocks        Used    Available Use% Mounted on\n"
           "/dev/vdb1      322122547200  1048576000 304836853760   1% /data\n")
CHECKS = ("check\tcategory\texit_code\n"
          "platform\trequired\t0\n"
          "gpu_identity\trequired\t0\n"
          "gpu_compute_cap\tdiagnostic\t0\n"
          "cpu_count\tdiagnostic\t0\n"
          "memory_bytes\tdiagnostic\t0\n"
          "docker_root_free\tdiagnostic\t0\n"
          "data_root_free\tdiagnostic\t0\n")


def texts(**overrides):
    base = {"checks.tsv": CHECKS, "gpu_identity.log": GPU_LOG, "gpu_compute_cap.log": CAP_LOG,
            "cpu_count.log": "15\n", "memory_bytes.log": FREE_LOG,
            "docker_root_free.log": DOCKER_DF, "data_root_free.log": DATA_DF}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def requirements(**overrides):
    base = {"gpu_name": WS, "gpu_count": 1, "min_vram_bytes": 100_000_000_000,
            "driver_branch": "580", "min_cuda_capability": "12.0",
            "min_guest_ram_bytes": 96_000_000_000, "min_vcpu": 8,
            "min_docker_data_root_free_bytes": 100_000_000_000,
            "min_data_root_free_bytes": 300_000_000_000}
    base.update(overrides)
    return base


def codes(result, severity=None):
    return sorted({i["code"] for i in result["issues"]
                   if severity is None or i["severity"] == severity})


def test_probe_files_are_the_probe_check_logs():
    assert PROBE_FILES == ("checks.tsv", "cpu_count.log", "data_root_free.log",
                           "docker_root_free.log", "gpu_compute_cap.log", "gpu_identity.log",
                           "memory_bytes.log")
    assert REQUIREMENT_FIELDS == (
        "gpu_name", "gpu_count", "min_vram_bytes", "driver_branch", "min_cuda_capability",
        "min_guest_ram_bytes", "min_vcpu", "min_docker_data_root_free_bytes",
        "min_data_root_free_bytes")


def test_known_host_fits_and_facts_are_the_literal_values():
    result = assess_host_fit(texts(), requirements())
    assert result["status"] == "passed", result["issues"]
    assert result["issues"] == [] and result["automatic_replacement"] is False
    assert result["facts"] == {
        "gpus": [{"index": 0, "name": WS, "driver_version": "580.95.05",
                  "memory_total_bytes": VRAM_BYTES, "compute_capability": "12.0"}],
        "vcpu": 15, "guest_ram_bytes": 104_152_227_840,
        "docker_data_root_available_bytes": 138_078_175_232,
        "data_root_available_bytes": 304_836_853_760}


def test_vram_boundary_equal_passes_one_byte_more_blocks():
    assert assess_host_fit(texts(), requirements(min_vram_bytes=VRAM_BYTES))["status"] == "passed"
    short = assess_host_fit(texts(), requirements(min_vram_bytes=VRAM_BYTES + 1))
    assert codes(short) == ["gpu_vram_short"]
    assert short["issues"][0]["path"] == "facts.gpus[0].memory_total_bytes"


@pytest.mark.parametrize("override, code", [
    ({"gpu_count": 2}, "gpu_count_short"),
    ({"gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}, "gpu_edition_mismatch"),
    ({"driver_branch": "570"}, "driver_branch_mismatch"),
    ({"min_cuda_capability": "12.1"}, "cuda_capability_short"),
    ({"min_guest_ram_bytes": 104_152_227_841}, "guest_ram_short"),
    ({"min_vcpu": 16}, "vcpu_short"),
    ({"min_docker_data_root_free_bytes": 138_078_175_233}, "docker_data_root_free_short"),
    ({"min_data_root_free_bytes": 304_836_853_761}, "data_root_free_short"),
])
def test_each_shortfall_is_its_own_blocking_issue(override, code):
    result = assess_host_fit(texts(), requirements(**override))
    assert result["status"] == "blocked"
    assert codes(result) == [code] and codes(result, "blocking") == [code]
    # Equal values pass: the same requirement minus the shortfall is satisfied.
    assert result["automatic_replacement"] is False


def test_capability_compares_numerically_not_as_text():
    # Text order would put "12.0" < "9.0" and "12.10" < "12.9".
    assert assess_host_fit(texts(), requirements(min_cuda_capability="9.0"))["status"] == "passed"
    cap = "index, compute_cap\n0, 12.10\n"
    assert assess_host_fit(texts(**{"gpu_compute_cap.log": cap}),
                           requirements(min_cuda_capability="12.9"))["status"] == "passed"
    cap = "index, compute_cap\n0, 8.9\n"
    assert codes(assess_host_fit(texts(**{"gpu_compute_cap.log": cap}), requirements())) == [
        "cuda_capability_short"]


def test_driver_family_is_the_major_version():
    log = GPU_LOG.replace("580.95.05", "590.10")
    result = assess_host_fit(texts(**{"gpu_identity.log": log}), requirements())
    assert codes(result) == ["driver_branch_mismatch"]
    log = GPU_LOG.replace("580.95.05", "580.126.09")
    assert assess_host_fit(texts(**{"gpu_identity.log": log}), requirements())[
        "status"] == "passed"


def test_two_gpus_every_one_is_checked():
    second = f"1, {WS.replace('Workstation', 'Server')}, GPU-x, 580.95.05, 97887 MiB, 0:08.0\n"
    caps = "index, compute_cap\n0, 12.0\n1, 12.0\n"
    result = assess_host_fit(texts(**{"gpu_identity.log": GPU_LOG + second,
                                      "gpu_compute_cap.log": caps}),
                             requirements(gpu_count=2))
    assert codes(result) == ["gpu_edition_mismatch"]
    assert result["issues"][0]["path"] == "facts.gpus[1].name"


def test_missing_compute_capability_log_is_unverified_never_a_pass():
    """The probe does not run the compute_cap query yet: every real host blocks on it."""
    checks = CHECKS.replace("gpu_compute_cap\tdiagnostic\t0\n", "")
    result = assess_host_fit(texts(**{"checks.tsv": checks, "gpu_compute_cap.log": None}),
                             requirements())
    assert result["status"] == "blocked"
    assert codes(result) == ["host_fact_unverified"]
    assert [i["path"] for i in result["issues"]] == ["facts.compute_capability"]
    assert result["facts"]["gpus"][0]["compute_capability"] is None
    assert result["facts"]["vcpu"] == 15


@pytest.mark.parametrize("change, path", [
    ({"checks.tsv": CHECKS.replace("cpu_count\tdiagnostic\t0", "cpu_count\tdiagnostic\t1")},
     "facts.vcpu"),
    ({"checks.tsv": CHECKS + "memory_bytes\tdiagnostic\t0\n"}, "facts.guest_ram_bytes"),
    ({"cpu_count.log": None}, "facts.vcpu"),
    ({"cpu_count.log": "fifteen\n"}, "facts.vcpu"),
    ({"cpu_count.log": "0\n"}, "facts.vcpu"),
    ({"gpu_identity.log": GPU_LOG.replace("97887 MiB", "97887 GB")}, "facts.gpus"),
    ({"gpu_identity.log": GPU_LOG.replace("index, name", "idx, name")}, "facts.gpus"),
    ({"gpu_identity.log": GPU_LOG.splitlines()[0] + "\n"}, "facts.gpus"),
    ({"memory_bytes.log": FREE_LOG.replace("Mem:", "Memory:")}, "facts.guest_ram_bytes"),
    ({"docker_root_free.log": DOCKER_DF + "/dev/sdb 1 1 1 1% /x\n"},
     "facts.docker_data_root_available_bytes"),
    ({"data_root_free.log": "df: /data: No such file or directory\n"},
     "facts.data_root_available_bytes"),
    ({"gpu_compute_cap.log": "index, compute_cap\n0, 12\n"}, "facts.compute_capability"),
    ({"gpu_compute_cap.log": "index, compute_cap\n3, 12.0\n"},
     "facts.gpus[0].compute_capability"),
])
def test_absent_failed_repeated_or_malformed_facts_are_unverified(change, path):
    result = assess_host_fit(texts(**change), requirements())
    assert result["status"] == "blocked"
    assert "host_fact_unverified" in codes(result)
    assert path in [i["path"] for i in result["issues"]]


def test_unusable_checks_table_trusts_no_log():
    facts, issues = parse_probe_facts(texts(**{"checks.tsv": None}))
    assert all(value is None for value in facts.values())
    assert {i["path"] for i in issues} == {
        "probe.checks.tsv", "facts.compute_capability", "facts.data_root_available_bytes",
        "facts.docker_data_root_available_bytes", "facts.gpus", "facts.guest_ram_bytes",
        "facts.vcpu"}


def test_wrapped_long_device_name_parses():
    wrapped = ("Filesystem        1B-blocks        Used    Available Use% Mounted on\n"
               "/dev/mapper/ubuntu--vg-ubuntu--lv--very--long--name\n"
               "               322122547200  1048576000 304836853760   1% /data\n")
    facts, issues = parse_probe_facts(texts(**{"data_root_free.log": wrapped}))
    assert issues == [] and facts["data_root_available_bytes"] == 304_836_853_760


@pytest.mark.parametrize("bad", [
    None, [], {}, dict(requirements(), extra=1),
    {k: v for k, v in requirements().items() if k != "min_vcpu"},
    requirements(gpu_name="TBD"), requirements(gpu_name=" padded "), requirements(gpu_name=""),
    requirements(gpu_count=0), requirements(gpu_count=True), requirements(min_vram_bytes=1.5),
    requirements(driver_branch="R580"), requirements(driver_branch=580),
    requirements(min_cuda_capability="12"), requirements(min_cuda_capability=12.0),
])
def test_invalid_requirements_fail_closed_without_comparing(bad):
    assert validate_requirements(bad)
    result = assess_host_fit(texts(), bad)
    assert result["status"] == "blocked"
    assert codes(result, "error") == ["invalid_requirements"]
    assert not set(codes(result)) - {"invalid_requirements"}


@pytest.mark.parametrize("garbage", [None, "text", 3, {"checks.tsv": 5},
                                     {"gpu_identity.log": ["x"]}])
def test_garbage_probe_texts_never_raise(garbage):
    result = assess_host_fit(garbage, requirements())
    assert result["status"] == "blocked" and "host_fact_unverified" in codes(result)


def test_report_header_is_deterministic_and_never_replaces_a_host():
    inputs = texts()
    before = deepcopy(inputs)
    first = host_fit_report(inputs, requirements(), scope="synthetic")
    assert first == host_fit_report(inputs, requirements(), scope="synthetic")
    assert inputs == before
    assert first["artifact_kind"] == "s2_host_fit_report" and first["schema_version"] == 1
    assert first["scope"] == "synthetic" and first["status"] == "passed"
    assert first["producer"]["module"] == "gate.s2_host_fit"
    assert first["inputs"] == {"probe_texts_sha256": identity_hash(inputs),
                               "requirements_sha256": identity_hash(requirements())}
    assert first["units"]["min_vram_bytes"] == "bytes" and first["units"]["vcpu"] == "count"
    assert first["automatic_replacement"] is False
    assert first["execution_status"] == "qualify_on_host"
    bad = host_fit_report(inputs, requirements(), scope="production")
    assert bad["scope"] is None and bad["status"] == "blocked"
    assert "invalid_scope" in codes(bad)
