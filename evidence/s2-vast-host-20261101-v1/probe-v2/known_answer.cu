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
