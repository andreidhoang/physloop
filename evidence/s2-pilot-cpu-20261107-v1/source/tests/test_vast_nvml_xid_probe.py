"""CPU-only fake-NVML contract checks; no real driver or GPU is accessed."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for fake-NVML test")
    source = Path(__file__).parents[1] / "scripts/vast_nvml_xid_probe.cpp"
    assert source.is_file()
    folder = tmp_path_factory.mktemp("fake-nvml")
    (folder / "nvml.h").write_text(r"""
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>
typedef int nvmlReturn_t;
enum { NVML_SUCCESS=0, NVML_ERROR_UNINITIALIZED=1, NVML_ERROR_INVALID_ARGUMENT=2,
       NVML_ERROR_NOT_SUPPORTED=3, NVML_ERROR_NO_PERMISSION=4, NVML_ERROR_NOT_FOUND=6,
       NVML_ERROR_DRIVER_NOT_LOADED=9, NVML_ERROR_TIMEOUT=10, NVML_ERROR_GPU_IS_LOST=15,
       NVML_ERROR_UNKNOWN=999 };
typedef void* nvmlDevice_t;
typedef void* nvmlEventSet_t;
const unsigned long long nvmlEventTypeXidCriticalError=8;
struct nvmlEventData_t { nvmlDevice_t device; unsigned long long eventType,eventData;
                        unsigned int gpuInstanceId,computeInstanceId; };
inline bool mode(const char* s) { return std::strcmp(std::getenv("NVML_SCENARIO"),s)==0; }
inline const char* nvmlErrorString(int) { return "fake NVML description"; }
inline int nvmlInit_v2() { return NVML_SUCCESS; }
inline int nvmlShutdown() { std::puts("fake_shutdown");return NVML_SUCCESS; }
inline int nvmlDeviceGetHandleByIndex_v2(unsigned int i,nvmlDevice_t* p) {
    if(i!=0)return NVML_ERROR_INVALID_ARGUMENT;*p=(void*)1;return NVML_SUCCESS;
}
inline int nvmlDeviceGetSupportedEventTypes(nvmlDevice_t,unsigned long long* p) {
    *p=mode("no_mask")?0:8;return NVML_SUCCESS;
}
inline int nvmlEventSetCreate(nvmlEventSet_t* p) { *p=(void*)2;return NVML_SUCCESS; }
inline int nvmlDeviceRegisterEvents(nvmlDevice_t,unsigned long long bits,nvmlEventSet_t) {
    if(bits!=8)return NVML_ERROR_INVALID_ARGUMENT;
    return mode("registration_failure")?NVML_ERROR_NOT_SUPPORTED:NVML_SUCCESS;
}
inline int nvmlEventSetWait_v2(nvmlEventSet_t,nvmlEventData_t* p,unsigned int timeout) {
    if(timeout!=100)return NVML_ERROR_INVALID_ARGUMENT;
    if(mode("wait_error"))return NVML_ERROR_UNKNOWN;
    if(mode("event")){p->eventType=8;p->eventData=31;return NVML_SUCCESS;}
    return NVML_ERROR_TIMEOUT;
}
inline int nvmlEventSetFree(nvmlEventSet_t) {
    std::puts("fake_event_set_freed");return NVML_SUCCESS;
}
""")
    binary = folder / "probe"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(folder), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return binary


@pytest.mark.parametrize(
    "scenario,exit_code,expected",
    [
        ("timeout", 0, "SUBSCRIPTION_AVAILABLE_DELIVERY_UNVERIFIED"),
        ("no_mask", 3, "XID_EVENT_NOT_ADVERTISED"),
        ("registration_failure", 2, "NVML_ERROR_NOT_SUPPORTED"),
        ("wait_error", 2, "NVML_ERROR_UNKNOWN"),
        ("event", 4, "NATURAL_XID_EVENT_OBSERVED"),
    ],
)
def test_noninvasive_subscription_outcomes(probe, scenario, exit_code, expected):
    result = subprocess.run(
        [str(probe)],
        env={**os.environ, "NVML_SCENARIO": scenario},
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == exit_code
    assert expected in result.stdout
    assert "fake_shutdown" in result.stdout
    assert ("fake_event_set_freed" in result.stdout) == (scenario != "no_mask")
    if scenario == "timeout":
        assert "NVML_ERROR_TIMEOUT" in result.stdout
        assert "zero_errors" not in result.stdout
