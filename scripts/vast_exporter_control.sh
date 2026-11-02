#!/usr/bin/env bash
# One temporary loopback-only public exporter and a synthetic load.
# This stops only the DCGM service installed for this disposable qualification VM.
set -u -o pipefail
umask 077
[[ $# == 2 && $1 == /* && $2 == /* ]] || exit 64
out=$1
work=$2
[[ -x $work/tensor_control ]] || exit 66
mkdir -m 700 "$out" || exit 73
cp "$0" "$out/"
name="physloop-dcgm-exporter-$$"
load_name="physloop-exporter-load-$$"
image='nvcr.io/nvidia/k8s/dcgm-exporter@sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b'
cuda_image='sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97'
load_pid=''
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    timeout -k 2s 8s docker logs "$name" > "$out/exporter.log" 2>&1 || true
    timeout -k 2s 8s docker rm -f "$name" "$load_name" > "$out/container-cleanup.log" 2>&1 || true
    if [[ -n $load_pid ]]; then wait "$load_pid" 2>/dev/null || true; fi
    printf 'script_exit=%s\nutc_end=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$out/manifest.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM
printf 'utc_start=%s\nexporter_image=%s\nmode=embedded; bundled DCGM 4.6.1 with exporter 4.8.4\n' "$(date -u +%FT%TZ)" "$image" > "$out/manifest.txt"
timeout -k 5s 180s docker pull "$image" > "$out/image-pull.log" 2>&1 || exit 2
timeout 10s docker image inspect "$image" --format '{{.Id}} {{json .RepoDigests}}' > "$out/image-identity.log" 2>&1 || exit 2
timeout 15s systemctl stop nvidia-dcgm > "$out/stop-task-host-engine.log" 2>&1 || exit 2
if pgrep -ax nv-hostengine > "$out/remaining-host-engine.log"; then exit 2; fi
timeout 20s docker run -d --rm --name "$name" --gpus all --cap-add SYS_ADMIN --network host "$image" -a 127.0.0.1:9400 -c 1000 -f /etc/dcgm-exporter/default-counters.csv > "$out/container-id.txt" 2>&1 || exit 2
ready=0
for attempt in $(seq 1 15); do
    if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9400/metrics > "$out/startup-$attempt.prom" 2> "$out/startup-$attempt.err"; then ready=1; break; fi
    sleep 1
done
[[ $ready == 1 ]] || exit 2
ss -ltnp > "$out/listeners.log" 2>&1
printf 'sample\tutc\tcurl_exit\n' > "$out/samples.tsv"
failed=0
for sample in $(seq 0 13); do
    if [[ $sample == 2 ]]; then
        timeout -k 5s 40s docker run --rm --name "$load_name" --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$cuda_image" timeout -k 5s 30s /probe/tensor_control > "$out/tensor-control.log" 2>&1 &
        load_pid=$!
    fi
    at=$(date -u +%FT%T.%NZ)
    curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9400/metrics > "$out/sample-$sample.prom" 2> "$out/sample-$sample.err"
    rc=$?
    printf '%s\t%s\t%s\n' "$sample" "$at" "$rc" >> "$out/samples.tsv"
    [[ $rc == 0 ]] || failed=1
    sleep 1
done
wait "$load_pid"
load_rc=$?
load_pid=''
printf 'load_exit=%s\n' "$load_rc" >> "$out/manifest.txt"
[[ $failed == 0 && $load_rc == 0 ]] || exit 2
printf 'SCRAPE_AND_LOAD_EXECUTION_PASS_REVIEW_ACTUAL_METRICS\n' > "$out/result.txt"
