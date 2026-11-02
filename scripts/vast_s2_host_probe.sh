#!/usr/bin/env bash
# Synthetic guest readiness only. No models, datasets, account or provider actions.
# See experiments/2026-11-01-vast-host-check.md before execution.
set -u -o pipefail
umask 077

if [[ $# != 1 || $1 != /* ]]; then
    echo 'Usage: bash vast_s2_host_probe.sh /absolute/new/evidence-directory' >&2
    exit 64
fi
command -v timeout >/dev/null || { echo 'GNU timeout is required.' >&2; exit 69; }
if [[ ${PHYSLOOP_PROBE_TIMEOUT_ACTIVE:-0} != 1 ]]; then
    PHYSLOOP_PROBE_TIMEOUT_ACTIVE=1 exec timeout --signal=TERM --kill-after=60s 20m bash "$0" "$@"
fi

out=$1
mkdir -m 700 -- "$out" || exit 73
work=$(mktemp -d /tmp/physloop-host-probe.XXXXXXXX) || exit 73
project="physloop-probe-$(date -u +%Y%m%d%H%M%S)-$$"
cuda_tag='nvidia/cuda:12.8.1-devel-ubuntu22.04'
docker=(docker)
compose_ready=0
failed=0
printf 'check\tcategory\texit_code\n' > "$out/checks.tsv"
printf 'UNFINISHED\n' > "$out/result.txt"
printf 'utc_start=%s\nproject=%s\npublic_image_tag=%s\n' "$(date -u +%FT%TZ)" "$project" "$cuda_tag" > "$out/manifest.txt"
cp -- "$0" "$out/probe-script.sh"
sha256sum "$out/probe-script.sh" > "$out/script.sha256"

check() {
    local category=$1 name=$2 seconds=$3 rc
    shift 3
    timeout --signal=TERM --kill-after=5s "${seconds}s" "$@" > "$out/$name.log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\n' "$name" "$category" "$rc" >> "$out/checks.tsv"
    printf '%s: %s (exit %s)\n' "$name" "$category" "$rc"
    if [[ $category == required && $rc != 0 ]]; then failed=1; fi
    return "$rc"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ $compose_ready == 1 ]]; then
        timeout -k 2s 6s "${docker[@]}" compose -p "$project" -f "$work/compose.yaml" logs --no-color > "$out/compose-services.log" 2>&1 || true
        timeout -k 2s 8s "${docker[@]}" compose -p "$project" -f "$work/compose.yaml" down --volumes --remove-orphans --timeout 2 > "$out/cleanup-compose.log" 2>&1 || true
    fi
    timeout -k 2s 4s "${docker[@]}" rm -f "$project-compile" "$project-cuda" "$project-vram" > "$out/cleanup-containers.log" 2>&1 || true
    timeout -k 2s 4s "${docker[@]}" image rm "$project:build-check" > "$out/cleanup-build-image.log" 2>&1 || true
    timeout -k 2s 4s "${docker[@]}" ps -aq --filter "label=com.docker.compose.project=$project" > "$out/remaining-compose-containers.log" 2>&1 || true
    # Preserve sources and any partial outputs before deleting only our workspace.
    cp -- "$work"/*.cu "$work"/Dockerfile "$work"/compose.yaml "$out/" 2>/dev/null || true
    rm -rf -- "$work"
    printf 'utc_end=%s\nscript_exit=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$out/manifest.txt"
    if [[ $rc != 0 ]]; then printf 'FAIL_OR_INTERRUPTED exit=%s; inspect checks and cleanup logs\n' "$rc" > "$out/result.txt"; fi
    printf 'Logs preserved: %s\nRental billing requires separate provider-side destruction.\n' "$out"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM

check required platform 10 bash -c 'test "$(uname -s)" = Linux && test "$(uname -m)" = x86_64' || true
check required os_identity 10 bash -c 'uname -srmo; cat /etc/os-release; printf "pid1="; cat /proc/1/comm; id -u' || true
check diagnostic systemd 10 systemctl is-system-running || true
check required guest_admin 10 bash -c 'test "$(id -u)" = 0 || sudo -n true' || true
check required gpu_identity 15 nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv || true
check diagnostic gpu_telemetry 15 nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv || true
check diagnostic nvml_reported_metrics 15 nvidia-smi -q -d POWER,TEMPERATURE,CLOCK,PERFORMANCE || true
if ! timeout -k 2s 8s docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    docker=(sudo -n docker)
fi
check required docker_engine 15 "${docker[@]}" version --format 'client={{.Client.Version}} server={{.Server.Version}}' || exit 2
check required compose_version 15 "${docker[@]}" compose version || exit 2
check required buildx_version 15 "${docker[@]}" buildx version || exit 2
check diagnostic docker_platform 10 "${docker[@]}" info --format 'os={{.OSType}} arch={{.Architecture}} driver={{.Driver}}' || true
check diagnostic disk_space 10 df -h "$work" || true
check required public_image_pull 600 "${docker[@]}" pull "$cuda_tag" || exit 2
check required image_identity 15 "${docker[@]}" image inspect "$cuda_tag" --format '{{.Id}} {{json .RepoDigests}} {{.Os}}/{{.Architecture}}' || exit 2
image_id=$(timeout -k 2s 10s "${docker[@]}" image inspect "$cuda_tag" --format '{{.Id}}') || exit 2
[[ $image_id =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
printf 'resolved_image_id=%s\n' "$image_id" >> "$out/manifest.txt"

cat > "$work/known_answer.cu" <<'CUDA'
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA error at %d: %s\n",__LINE__,cudaGetErrorString(e)); return 2; } } while(0)
constexpr size_t N=1048576;
uint32_t reference(uint32_t i) { uint64_t wide=uint64_t(i)*1664525ull+1013904223ull; return uint32_t(wide&0xffffffffull)^(i/8); }
__global__ void known_answer(uint32_t* out) { uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) out[i]=(i*1664525u+1013904223u)^(i>>3); }
size_t mismatch(const std::vector<uint32_t>& a) { size_t n=0; for(size_t i=0;i<N;i++) n+=(a[i]!=reference(uint32_t(i))); return n; }
__global__ void fill_vram(unsigned long long* p,size_t words) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)
        p[i]=static_cast<unsigned long long>(i)^0x9e3779b97f4a7c15ull;
}
__global__ void verify_vram(const unsigned long long* p,size_t words,unsigned long long* mismatches) {
    unsigned long long bad=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)
        bad+=(p[i]!=(static_cast<unsigned long long>(i)^0x9e3779b97f4a7c15ull));
    if(bad)atomicAdd(mismatches,bad);
}
int vram_check() {
    CUDA_CHECK(cudaSetDevice(0));size_t free_bytes=0,total_bytes=0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    constexpr size_t GiB=size_t(1)<<30;
    std::printf("vram_total_bytes=%zu vram_free_before_bytes=%zu tested_device=0\n",total_bytes,free_bytes);std::fflush(stdout);
    if(free_bytes<8*GiB){std::fprintf(stderr,"Insufficient free memory for this check (requires 8 GiB).\n");return 20;}
    size_t bytes=(free_bytes/4)*3;if(bytes>72*GiB)bytes=72*GiB;bytes-=bytes%sizeof(unsigned long long);
    const size_t words=bytes/sizeof(unsigned long long);
    unsigned long long *p=nullptr,*bad_device=nullptr,bad=0;
    CUDA_CHECK(cudaMalloc(&p,bytes));CUDA_CHECK(cudaMalloc(&bad_device,sizeof(bad)));CUDA_CHECK(cudaMemset(bad_device,0,sizeof(bad)));
    std::printf("vram_allocated_bytes=%zu full_gpu_verification_words=%zu cpu_samples=1024\n",bytes,words);std::fflush(stdout);
    fill_vram<<<1024,256>>>(p,words);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    verify_vram<<<1024,256>>>(p,words,bad_device);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&bad,bad_device,sizeof(bad),cudaMemcpyDeviceToHost));
    size_t sample_bad=0;
    for(size_t s=0;s<1024;s++){
        size_t i=(words-1)*s/1023;unsigned long long actual=0;
        CUDA_CHECK(cudaMemcpy(&actual,p+i,sizeof(actual),cudaMemcpyDeviceToHost));
        const unsigned long long expected=(~static_cast<unsigned long long>(i))^(~0x9e3779b97f4a7c15ull);
        sample_bad+=(actual!=expected);
    }
    CUDA_CHECK(cudaFree(bad_device));CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    std::printf("vram_gpu_mismatches=%llu vram_cpu_sample_mismatches=%zu vram_free_after_bytes=%zu allocation_released=PASS\n",bad,sample_bad,free_bytes);
    return (bad || sample_bad)?21:0;
}
bool send_text(int fd,const char* text) { size_t n=std::strlen(text); while(n) { ssize_t k=send(fd,text,n,MSG_NOSIGNAL); if(k<=0)return false; text+=k;n-=k; } return true; }
std::string read_line(int fd) { std::string s; char c; while(s.size()<128 && recv(fd,&c,1,0)==1) { s+=c; if(c=='\n')break; } return s; }
int server() {
    std::ofstream("/probe/shared-marker.txt") << "physloop-shared-marker\n";
    int fd=socket(AF_INET,SOCK_STREAM,0); if(fd<0)return 10;
    int yes=1; setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(9099);
    if(bind(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a)) || listen(fd,4))return 11;
    for(;;) { int c=accept(fd,nullptr,nullptr); if(c<0)return 12; bool ok=read_line(c)=="physloop-request\n" && send_text(c,"physloop-response\n");close(c);if(!ok)return 13; }
}
int client() {
    for(int attempt=0;attempt<50;attempt++) {
        addrinfo hints{},*res=nullptr; hints.ai_family=AF_INET;hints.ai_socktype=SOCK_STREAM;
        if(getaddrinfo("server","9099",&hints,&res)==0) {
            int fd=socket(res->ai_family,res->ai_socktype,res->ai_protocol);
            bool connected=fd>=0 && connect(fd,res->ai_addr,res->ai_addrlen)==0;freeaddrinfo(res);
            if(connected) {
                bool ok=send_text(fd,"physloop-request\n") && read_line(fd)=="physloop-response\n";close(fd);
                std::ifstream f("/probe/shared-marker.txt");std::string s;std::getline(f,s);
                if(!ok || s!="physloop-shared-marker")return 14;
                std::puts("compose_dns_network_shared_mount=PASS");return 0;
            }
            if(fd>=0)close(fd);
        }
        usleep(200000);
    }
    return 15;
}
int main(int argc,char** argv) {
    if(argc==2 && std::strcmp(argv[1],"vram")==0)return vram_check();
    if(argc==2 && std::strcmp(argv[1],"server")==0)return server();
    if(argc==2 && std::strcmp(argv[1],"client")==0)return client();
    if(reference(0)!=1013904223u)return 3;
    int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));if(count<1)return 4;
    CUDA_CHECK(cudaSetDevice(0));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::printf("visible_devices=%d tested_device=0 name=%s elements=%zu\n",count,prop.name,N);
    uint32_t* device=nullptr;CUDA_CHECK(cudaMalloc(&device,N*sizeof(uint32_t)));
    std::vector<uint32_t> got(N),first(N);
    for(int repeat=0;repeat<3;repeat++) {
        known_answer<<<(N+255)/256,256>>>(device);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(),device,N*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        size_t n=mismatch(got);if(n){std::printf("mismatches=%zu\n",n);return 5;}
        if(repeat==0)first=got;else if(std::memcmp(first.data(),got.data(),N*sizeof(uint32_t)))return 6;
    }
    got[0]^=1;if(mismatch(got)!=1)return 7;
    CUDA_CHECK(cudaFree(device));
    std::puts("cpu_reference_mismatches=0 repeat_bitwise=PASS negative_control_mismatches=1");return 0;
}
CUDA

cat > "$work/Dockerfile" <<DOCKERFILE
FROM $cuda_tag
RUN test -x /usr/local/cuda/bin/nvcc
DOCKERFILE
# A recorded local tag is used only for FROM. Its image ID was recorded above;
# compilation and all execution use that immutable local image ID directly.
check required buildx_build 120 "${docker[@]}" buildx build --load --pull=false --network=none --tag "$project:build-check" "$work" || true
check required cuda_compile 120 "${docker[@]}" run --rm --name "$project-compile" --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$image_id" bash -c 'nvcc -O2 -arch=native /probe/known_answer.cu -o /probe/known_answer' || exit 2
check required cuda_known_answer 60 "${docker[@]}" run --rm --name "$project-cuda" --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$image_id" /probe/known_answer || true
if ! check required cuda_vram_integrity 60 "${docker[@]}" run --rm --name "$project-vram" --gpus all --network none --mount "type=bind,src=$work,dst=/probe" "$image_id" timeout --signal=TERM --kill-after=5s 30s /probe/known_answer vram; then
    check diagnostic vram_failure_cleanup 10 "${docker[@]}" rm -f "$project-vram" || true
fi

export PHYSLOOP_PROBE_IMAGE="$image_id" PHYSLOOP_PROBE_WORK="$work"
export PHYSLOOP_EXPECT_PIDNS
PHYSLOOP_EXPECT_PIDNS=$(readlink /proc/self/ns/pid)
cat > "$work/compose.yaml" <<'COMPOSE'
services:
  server:
    image: ${PHYSLOOP_PROBE_IMAGE}
    pull_policy: never
    command: ["/probe/known_answer", "server"]
    volumes: ["${PHYSLOOP_PROBE_WORK}:/probe"]
    networks: [probe]
  client:
    image: ${PHYSLOOP_PROBE_IMAGE}
    pull_policy: never
    pid: host
    command:
      - bash
      - -ec
      - 'test "$$(readlink /proc/self/ns/pid)" = "$$PHYSLOOP_EXPECT_PIDNS"; echo guest_host_pid_namespace=PASS; /probe/known_answer client; /probe/known_answer'
    environment:
      PHYSLOOP_EXPECT_PIDNS: ${PHYSLOOP_EXPECT_PIDNS}
    volumes: ["${PHYSLOOP_PROBE_WORK}:/probe"]
    networks: [probe]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
networks:
  probe:
    driver: bridge
    internal: true
COMPOSE
compose_ready=1
check required compose_config 15 "${docker[@]}" compose -p "$project" -f "$work/compose.yaml" config --quiet || true
check required compose_execution 90 "${docker[@]}" compose -p "$project" -f "$work/compose.yaml" up --abort-on-container-exit --exit-code-from client --timeout 3 || true

check diagnostic profiler_policy 10 bash -c 'if test -r /proc/driver/nvidia/params; then grep -E "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params; else echo UNVERIFIED_no_readable_driver_policy; exit 1; fi' || true
check diagnostic process_capabilities 10 bash -c 'grep -E "^Cap(Eff|Prm|Bnd):" /proc/self/status; test ! -r /proc/sys/kernel/perf_event_paranoid || cat /proc/sys/kernel/perf_event_paranoid' || true
monitor=()
if [[ $(id -u) != 0 ]] && timeout 5s sudo -n true; then monitor=(sudo -n); fi
if command -v nsys >/dev/null; then
    check diagnostic nsys_version 10 nsys --version || true
    check diagnostic nsys_environment 15 nsys status -e || true
    check diagnostic nsys_cuda_trace 90 "${monitor[@]}" "$(command -v nsys)" profile --sample=none --cpuctxsw=none --trace=cuda,nvtx -o "$out/nsys-cuda" "$work/known_answer" || true
else
    printf 'UNVERIFIED: nsys is absent; no CUDA/NVTX trace or counter test performed.\n' > "$out/nsys-unverified.log"
fi
if command -v ncu >/dev/null; then
    check diagnostic ncu_version 10 ncu --version || true
    check diagnostic ncu_actual_kernel 90 "${monitor[@]}" "$(command -v ncu)" --set basic --launch-count 1 "$work/known_answer" || true
else
    printf 'UNVERIFIED: host ncu is absent; hardware counter support is not established, not disproved. Public profiler installation is an operator setup decision.\n' > "$out/ncu-unverified.log"
fi
if command -v dcgmi >/dev/null; then
    check diagnostic dcgm_discovery 15 "${monitor[@]}" dcgmi discovery -l || true
    check diagnostic dcgm_profile_fields 15 "${monitor[@]}" dcgmi profile --list -i 0 || true
    check diagnostic dcgm_tensor_active 15 "${monitor[@]}" dcgmi dmon -e 1004 -i 0 -c 3 -d 1000 || true
else
    printf 'UNVERIFIED: dcgmi is absent; tensor-active/energy/XID/PCIe field support not established.\n' > "$out/dcgm-unverified.log"
fi
printf 'No driver/service changes made. Review actual counter samples and any Nsight results; tool absence does not establish hardware incompatibility. Exporter operation and the complete S2 field set are not tested. N/A is not zero. No scientific S2 admission.\n' > "$out/monitoring-review.txt"
if [[ $failed != 0 ]]; then exit 2; fi
printf 'DEPLOYMENT_PASS_MONITORING_REVIEW_REQUIRED\n' > "$out/result.txt"
exit 0
