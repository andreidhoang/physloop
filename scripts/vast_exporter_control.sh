#!/usr/bin/env bash
# One temporary loopback-only public exporter and a synthetic load.
# Refuses existing host engines; never changes host-service state or pulls images.
# Usage: vast_exporter_control.sh OUT WORK CACHED_CUDA_REPOSITORY@sha256:DIGEST
# Both the explicit CUDA image and the pinned exporter below must already be cached.
set -u -o pipefail
umask 077
[[ $# == 3 && $1 == /* && $2 == /* ]] || exit 64
out=$1
work=$2
cuda_ref=$3
[[ $cuda_ref =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}$ ]] || exit 64
[[ -x $work/tensor_control ]] || exit 66
mkdir -m 700 "$out" || exit 73
cp "$0" "$out/"
name="physloop-dcgm-exporter-$$"
load_name="physloop-exporter-load-$$"
image_ref='nvcr.io/nvidia/k8s/dcgm-exporter@sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b'
resolve_image() {
    local ref=$1 log=$2 id digest matched=0
    timeout -k 2s 10s docker image inspect "$ref" \
        --format '{{.Id}}{{range .RepoDigests}}{{printf "\n%s" .}}{{end}}' > "$log" 2> "$log.err" || return 2
    { IFS= read -r id || return 2
      [[ $id =~ ^sha256:[0-9a-f]{64}$ ]] || return 2
      while IFS= read -r digest; do [[ $digest == "$ref" ]] && matched=1; done
    } < "$log"
    [[ $matched == 1 ]] || return 2
    printf '%s\n' "$id"
}
printf 'utc_start=%s\nexporter_ref=%s\ncuda_ref=%s\n' \
    "$(date -u +%FT%TZ)" "$image_ref" "$cuda_ref" > "$out/manifest.txt"
image=$(resolve_image "$image_ref" "$out/image-identity.log") || exit 2
cuda_image=$(resolve_image "$cuda_ref" "$out/cuda-image-identity.log") || exit 2
printf 'exporter_image_id=%s\ncuda_image_id=%s\n' "$image" "$cuda_image" >> "$out/manifest.txt"
timeout -k 2s 10s systemctl is-active --quiet nvidia-dcgm > "$out/host-engine-service.log" 2>&1
service_rc=$?
printf 'host_engine_service_exit=%s\n' "$service_rc" >> "$out/manifest.txt"
# systemctl: 3 means inactive, 4 means no such service. Other results are unsafe/unknown.
[[ $service_rc == 3 || $service_rc == 4 ]] || exit 2
timeout -k 2s 10s pgrep -ax nv-hostengine > "$out/remaining-host-engine.log" 2>&1
engine_rc=$?
printf 'host_engine_process_exit=%s\n' "$engine_rc" >> "$out/manifest.txt"
[[ $engine_rc == 1 ]] || exit 2
# A healthy foreign endpoint must not be mistaken for this probe's exporter.
timeout -k 2s 10s ss -H -ltn 'sport = :9400' > "$out/listeners-before.log" 2>&1 || exit 2
[[ ! -s $out/listeners-before.log ]] || exit 2
owner_receipt=$(mktemp "$out/owner.XXXXXXXXXXXX") || exit 2
owner_token=${owner_receipt##*/}
printf 'container_owner=%s\n' "$owner_token" >> "$out/manifest.txt"
name_available() {
    timeout -k 2s 10s docker container ls --all --filter "name=^/$1$" \
        --format '{{.ID}}' > "$2" 2> "$2.err" || return 2
    [[ ! -s $2 ]]
}
remove_owned() {
    local cidfile=$1 expected_name=$2 cid present found_id found_name found_owner
    [[ -f $cidfile && ! -L $cidfile ]] || return 0
    cid=$(cat "$cidfile") || return 2
    [[ $cid =~ ^[0-9a-f]{64}$ ]] || return 2
    timeout -k 2s 10s docker container ls --all --no-trunc --filter "id=$cid" \
        --format '{{.ID}}' > "$cidfile.inventory" 2> "$cidfile.inventory.err" || return 2
    present=$(cat "$cidfile.inventory")
    [[ -n $present ]] || return 0
    [[ $present == "$cid" ]] || return 2
    timeout -k 2s 10s docker container inspect "$cid" \
        --format '{{.Id}}{{printf "\n%s\n" .Name}}{{index .Config.Labels "physloop.probe_owner"}}' \
        > "$cidfile.identity" 2> "$cidfile.identity.err" || return 2
    { IFS= read -r found_id && IFS= read -r found_name && IFS= read -r found_owner; } \
        < "$cidfile.identity" || return 2
    [[ $found_id == "$cid" && $found_name == "/$expected_name" && $found_owner == "$owner_token" ]] || return 2
    timeout -k 2s 8s docker logs "$cid" > "$cidfile.container.log" 2>&1 || true
    timeout -k 2s 10s docker rm -f "$cid" > "$cidfile.cleanup.log" 2>&1
}
verify_exporter_endpoint() {
    local phase=$1 cid found_id found_name found_owner running pid pids=''
    local line state recv send address peer processes count=0 has_pid
    [[ -f $out/exporter.cid && ! -L $out/exporter.cid ]] || return 2
    cid=$(cat "$out/exporter.cid") || return 2
    [[ $cid =~ ^[0-9a-f]{64}$ ]] || return 2
    timeout -k 2s 10s docker container inspect "$cid" \
        --format '{{.Id}}{{printf "\n%s\n" .Name}}{{index .Config.Labels "physloop.probe_owner"}}{{printf "\n%t" .State.Running}}' \
        > "$out/$phase-exporter-identity.log" 2> "$out/$phase-exporter-identity.err" || return 2
    { IFS= read -r found_id && IFS= read -r found_name && IFS= read -r found_owner && IFS= read -r running; } \
        < "$out/$phase-exporter-identity.log" || return 2
    [[ $found_id == "$cid" && $found_name == "/$name" && $found_owner == "$owner_token" && $running == true ]] || return 2
    timeout -k 2s 10s docker top "$cid" -eo pid \
        > "$out/$phase-exporter-pids.log" 2> "$out/$phase-exporter-pids.err" || return 2
    while IFS= read -r line; do
        [[ $line =~ ^[[:space:]]*PID[[:space:]]*$ ]] && continue
        [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]] || return 2
        pids="$pids ${BASH_REMATCH[1]}"
    done < "$out/$phase-exporter-pids.log"
    [[ -n $pids ]] || return 2
    timeout -k 2s 10s ss -H -ltnp 'sport = :9400' \
        > "$out/$phase-listeners.log" 2> "$out/$phase-listeners.err" || return 2
    while IFS= read -r line; do
        read -r state recv send address peer processes <<< "$line"
        [[ $state == LISTEN && $address == 127.0.0.1:9400 ]] || return 2
        count=$((count + 1))
        has_pid=0
        while [[ $processes =~ pid=([0-9]+), ]]; do
            pid=${BASH_REMATCH[1]}
            [[ "$pids " == *" $pid "* ]] || return 2
            processes=${processes#*"pid=$pid,"}
            has_pid=1
        done
        [[ $has_pid == 1 ]] || return 2
    done < "$out/$phase-listeners.log"
    [[ $count == 1 ]]
}
name_available "$name" "$out/exporter-name-inventory.log" || exit 2
name_available "$load_name" "$out/load-name-inventory.log" || exit 2
load_pid=''
cleanup() {
    local rc=$? cleanup_rc=0
    trap - EXIT INT TERM
    if [[ -n $load_pid ]]; then wait "$load_pid" 2>/dev/null || true; fi
    remove_owned "$out/load.cid" "$load_name" || cleanup_rc=2
    remove_owned "$out/exporter.cid" "$name" || cleanup_rc=2
    [[ $rc != 0 || $cleanup_rc == 0 ]] || rc=2
    printf 'cleanup_exit=%s\n' "$cleanup_rc" >> "$out/manifest.txt"
    printf 'script_exit=%s\nutc_end=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$out/manifest.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM
timeout -k 2s 20s docker run --pull=never -d --rm --name "$name" \
    --cidfile "$out/exporter.cid" --label "physloop.probe_owner=$owner_token" \
    --gpus all --cap-add SYS_ADMIN --network host "$image" -a 127.0.0.1:9400 -c 1000 \
    -f /etc/dcgm-exporter/default-counters.csv > "$out/container-id.txt" 2>&1 || exit 2
[[ -s $out/exporter.cid ]] || exit 2
ready=0
for attempt in $(seq 1 15); do
    if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9400/metrics > "$out/startup-$attempt.prom" 2> "$out/startup-$attempt.err"; then
        verify_exporter_endpoint "startup-$attempt" || exit 2
        ready=1
        break
    fi
    sleep 1
done
[[ $ready == 1 ]] || exit 2
printf 'sample\tutc\tcurl_exit\n' > "$out/samples.tsv"
failed=0
for sample in $(seq 0 13); do
    if [[ $sample == 2 ]]; then
        timeout -k 5s 40s docker run --pull=never --rm --name "$load_name" \
            --cidfile "$out/load.cid" --label "physloop.probe_owner=$owner_token" \
            --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$cuda_image" \
            timeout -k 5s 30s /probe/tensor_control > "$out/tensor-control.log" 2>&1 &
        load_pid=$!
    fi
    at=$(date -u +%FT%T.%NZ)
    curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9400/metrics > "$out/sample-$sample.prom" 2> "$out/sample-$sample.err"
    rc=$?
    printf '%s\t%s\t%s\n' "$sample" "$at" "$rc" >> "$out/samples.tsv"
    [[ $rc == 0 ]] || failed=1
    verify_exporter_endpoint "sample-$sample" || exit 2
    sleep 1
done
wait "$load_pid"
load_rc=$?
load_pid=''
printf 'load_exit=%s\n' "$load_rc" >> "$out/manifest.txt"
[[ $failed == 0 && $load_rc == 0 && -s $out/load.cid ]] || exit 2
verify_exporter_endpoint final || exit 2
printf 'SCRAPE_AND_LOAD_EXECUTION_PASS_REVIEW_ACTUAL_METRICS\n' > "$out/result.txt"
