#!/usr/bin/env bash
# Run after Nsight has finished; DCGM and Nsight can contend for counters.
# Public synthetic inputs only. No provider actions or package installation.
set -u -o pipefail
umask 077
[[ $# == 2 && $1 == /* && $2 == /* ]] || exit 64
out=$1
work=$2
[[ -x $work/tensor_control && -f $work/vast_tensor_control.cu ]] || exit 66
mkdir -m 700 "$out" || exit 73
cp "$0" "$work/vast_tensor_control.cu" "$out/"
image='sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97'
name="physloop-tensor-control-$$"
monitor_pid=''
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n $monitor_pid ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    timeout -k 2s 10s docker rm -f "$name" > "$out/container-cleanup.log" 2>&1 || true
    printf 'script_exit=%s\nutc_end=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$out/manifest.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM
printf 'utc_start=%s\nimage_id=%s\n' "$(date -u +%FT%TZ)" "$image" > "$out/manifest.txt"
timeout -k 2s 10s dcgmi dmon -l > "$out/field-catalog.log" 2>&1
catalog_rc=$?
timeout -k 2s 20s dcgmi dmon -e 1004,1002,1005,150,155,156,203,230,100,101,1009,1010 -i 0 -c 14 -d 1000 > "$out/dcgm-samples.log" 2>&1 &
monitor_pid=$!
sleep 2
timeout -k 5s 40s docker run --rm --name "$name" --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$image" timeout -k 5s 30s /probe/tensor_control > "$out/tensor-control.log" 2>&1
tensor_rc=$?
wait "$monitor_pid"
monitor_rc=$?
monitor_pid=''
timeout -k 2s 10s nvidia-smi --query-gpu=name,driver_version,power.draw,temperature.gpu,clocks.sm,memory.used --format=csv > "$out/nvml-after.log" 2>&1
nvml_rc=$?
printf 'catalog_exit=%s\ntensor_exit=%s\nmonitor_exit=%s\nnvml_exit=%s\n' "$catalog_rc" "$tensor_rc" "$monitor_rc" "$nvml_rc" >> "$out/manifest.txt"
if [[ $tensor_rc != 0 || $monitor_rc != 0 || $catalog_rc != 0 || $nvml_rc != 0 ]]; then exit 2; fi
printf 'EXECUTION_PASS_REVIEW_ACTUAL_COUNTER_VALUES\n' > "$out/result.txt"
