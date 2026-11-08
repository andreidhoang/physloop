// Agent-authored S2/G7 non-invasive XID subscription diagnostic for GPU 0.
// Compile in the public CUDA devel image (stub is link-time only):
// g++ -std=c++17 -I/usr/local/cuda/include vast_nvml_xid_probe.cpp \
//     -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o nvml_xid_probe
// Run with the actual driver library, never the stub, and a 10-second outer timeout.
// No injection, reset, clock, power, persistence or provider operations.
#include <nvml.h>
#include <cstdio>

static const char* status_name(nvmlReturn_t status) {
    switch (status) {
        case NVML_SUCCESS: return "NVML_SUCCESS";
        case NVML_ERROR_UNINITIALIZED: return "NVML_ERROR_UNINITIALIZED";
        case NVML_ERROR_INVALID_ARGUMENT: return "NVML_ERROR_INVALID_ARGUMENT";
        case NVML_ERROR_NOT_SUPPORTED: return "NVML_ERROR_NOT_SUPPORTED";
        case NVML_ERROR_NO_PERMISSION: return "NVML_ERROR_NO_PERMISSION";
        case NVML_ERROR_NOT_FOUND: return "NVML_ERROR_NOT_FOUND";
        case NVML_ERROR_DRIVER_NOT_LOADED: return "NVML_ERROR_DRIVER_NOT_LOADED";
        case NVML_ERROR_TIMEOUT: return "NVML_ERROR_TIMEOUT";
        case NVML_ERROR_GPU_IS_LOST: return "NVML_ERROR_GPU_IS_LOST";
        case NVML_ERROR_UNKNOWN: return "NVML_ERROR_UNKNOWN";
        default: return "NVML_OTHER_STATUS_SEE_CODE_AND_DESCRIPTION";
    }
}

static bool report(const char* operation, nvmlReturn_t status) {
    std::printf("operation=%s status=%s code=%d description=%s\n",
                operation, status_name(status), int(status), nvmlErrorString(status));
    std::fflush(stdout);
    return status == NVML_SUCCESS;
}

static int probe(nvmlDevice_t device) {
    unsigned long long supported = 0;
    if (!report("nvmlDeviceGetSupportedEventTypes",
                nvmlDeviceGetSupportedEventTypes(device, &supported))) return 2;
    std::printf("gpu_index=0 supported_events_mask=0x%llx xid_bit=0x%llx\n",
                supported, static_cast<unsigned long long>(nvmlEventTypeXidCriticalError));
    if (!(supported & nvmlEventTypeXidCriticalError)) {
        std::puts("XID_EVENT_NOT_ADVERTISED; delivery=UNVERIFIED; not a zero-error result");
        return 3;
    }
    nvmlEventSet_t set{};
    if (!report("nvmlEventSetCreate", nvmlEventSetCreate(&set))) return 2;
    int result = 2;
    if (report("nvmlDeviceRegisterEvents",
               nvmlDeviceRegisterEvents(device, nvmlEventTypeXidCriticalError, set))) {
        nvmlEventData_t event{};
        const auto status = nvmlEventSetWait_v2(set, &event, 100);
        report("nvmlEventSetWait_v2_timeout_100ms", status);
        if (status == NVML_ERROR_TIMEOUT) {
            std::puts("SUBSCRIPTION_AVAILABLE_DELIVERY_UNVERIFIED; "
                      "no event received during short wait; not a zero-error result");
            result = 0;
        } else if (status == NVML_SUCCESS
                   && (event.eventType & nvmlEventTypeXidCriticalError)) {
            std::printf("NATURAL_XID_EVENT_OBSERVED event_type=0x%llx xid_code=%llu; "
                        "no fault was injected; operator review required\n",
                        event.eventType, event.eventData);
            result = 4;
        } else {
            std::puts("SUBSCRIPTION_AVAILABLE_WAIT_INCONCLUSIVE; delivery=UNVERIFIED");
        }
    } else {
        std::puts("SUBSCRIPTION_UNVERIFIED; delivery=UNVERIFIED");
    }
    if (!report("nvmlEventSetFree", nvmlEventSetFree(set))) result = 2;
    return result;
}

int main() {
    std::puts("NVML XID subscription check only; no DCGM delivery or hardware-health qualification");
    if (!report("nvmlInit_v2", nvmlInit_v2())) return 2;
    nvmlDevice_t device{};
    int result = 2;
    if (report("nvmlDeviceGetHandleByIndex_v2", nvmlDeviceGetHandleByIndex_v2(0, &device))) {
        result = probe(device);
    }
    if (!report("nvmlShutdown", nvmlShutdown())) result = 2;
    return result;
}
