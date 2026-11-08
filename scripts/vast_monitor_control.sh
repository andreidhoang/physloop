#!/usr/bin/env bash
# Run after Nsight has finished; DCGM and Nsight can contend for counters.
# Public synthetic inputs only. No provider actions, image pulls or package installation.
# Usage: vast_monitor_control.sh OUT WORK CACHED_REPOSITORY@sha256:DIGEST
set -u -o pipefail
umask 077
[[ $# == 3 && $1 == /* && $2 == /* ]] || exit 64
out=$1
work=$2
image_ref=$3
[[ $image_ref =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}$ ]] || exit 64
[[ -x $work/tensor_control && -f $work/vast_tensor_control.cu ]] || exit 66
mkdir -m 700 "$out" || exit 73
cp "$0" "$work/vast_tensor_control.cu" "$out/"
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
printf 'utc_start=%s\nimage_ref=%s\n' "$(date -u +%FT%TZ)" "$image_ref" > "$out/manifest.txt"
image=$(resolve_image "$image_ref" "$out/image-identity.log") || exit 2
printf 'image_id=%s\n' "$image" >> "$out/manifest.txt"
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
    [[ -n $present ]] || return 0  # A --rm container may already be absent.
    [[ $present == "$cid" ]] || return 2
    timeout -k 2s 10s docker container inspect "$cid" \
        --format '{{.Id}}{{printf "\n%s\n" .Name}}{{index .Config.Labels "physloop.probe_owner"}}' \
        > "$cidfile.identity" 2> "$cidfile.identity.err" || return 2
    { IFS= read -r found_id && IFS= read -r found_name && IFS= read -r found_owner; } \
        < "$cidfile.identity" || return 2
    [[ $found_id == "$cid" && $found_name == "/$expected_name" && $found_owner == "$owner_token" ]] || return 2
    timeout -k 2s 10s docker rm -f "$cid" > "$cidfile.cleanup.log" 2>&1
}
name="physloop-tensor-control-$$"
name_available "$name" "$out/name-inventory.log" || exit 2
monitor_pid=''
cleanup() {
    local rc=$? cleanup_rc=0
    trap - EXIT INT TERM
    if [[ -n $monitor_pid ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    remove_owned "$out/load.cid" "$name" || cleanup_rc=2
    [[ $rc != 0 || $cleanup_rc == 0 ]] || rc=2
    printf 'cleanup_exit=%s\n' "$cleanup_rc" >> "$out/manifest.txt"
    printf 'script_exit=%s\nutc_end=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$out/manifest.txt"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM
timeout -k 2s 10s dcgmi dmon -l > "$out/field-catalog.log" 2>&1
catalog_rc=$?
timeout -k 2s 20s dcgmi dmon -e 1004,1002,1005,150,155,156,203,230,100,101,1009,1010 -i 0 -c 14 -d 1000 > "$out/dcgm-samples.log" 2>&1 &
monitor_pid=$!
sleep 2
timeout -k 5s 40s docker run --pull=never --rm --name "$name" \
    --cidfile "$out/load.cid" --label "physloop.probe_owner=$owner_token" \
    --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$image" \
    timeout -k 5s 30s /probe/tensor_control > "$out/tensor-control.log" 2>&1
tensor_rc=$?
[[ $tensor_rc != 0 || -s $out/load.cid ]] || tensor_rc=2
wait "$monitor_pid"
monitor_rc=$?
monitor_pid=''
timeout -k 2s 10s nvidia-smi --query-gpu=name,driver_version,power.draw,temperature.gpu,clocks.sm,memory.used --format=csv > "$out/nvml-after.log" 2>&1
nvml_rc=$?
printf 'catalog_exit=%s\ntensor_exit=%s\nmonitor_exit=%s\nnvml_exit=%s\n' "$catalog_rc" "$tensor_rc" "$monitor_rc" "$nvml_rc" >> "$out/manifest.txt"
if [[ $tensor_rc != 0 || $monitor_rc != 0 || $catalog_rc != 0 || $nvml_rc != 0 ]]; then exit 2; fi
printf 'EXECUTION_PASS_REVIEW_ACTUAL_COUNTER_VALUES\n' > "$out/result.txt"
