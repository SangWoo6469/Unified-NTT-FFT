#include <vector>
#include <random>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include "config.hpp"
#include "types.hpp"
#include "reference_cpu.hpp"

// extern
void precompute_and_upload_all();
void make_plan_and_offsets(int*& d_plan, int*& d_offsets);
void launch_forward_bfu(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch);

// ----- 간단 CPU INTT(오라클) -----
// gamma = primitive 2Nth root, odd-exponent 스케줄의 역변환(+ n^{-1} mod q)
static uint32_t mod_pow_u(uint64_t a, uint64_t e){
    uint64_t r=1%Q; a%=Q; while(e){ if(e&1) r=(r*a)%Q; a=(a*a)%Q; e>>=1; } return (uint32_t)r;
}
static int bitlen(int n){ int b=0; while((1<<b)<n) ++b; return b; }
static int bitrev(int x, int b){ int r=0; for(int i=0;i<b;++i){ r=(r<<1)|(x&1); x>>=1; } return r; }

static void cpu_intt_inverse(const std::vector<uint32_t>& A, std::vector<uint32_t>& a){
    int n=(int)A.size(); a=A;
    // 입력 비트리버스
    int B=bitlen(n);
    for(int i=0;i<n;++i){ int j=bitrev(i,B); if(j>i) std::swap(a[i],a[j]); }
    // gamma^{-1}
    uint32_t g=3;
    uint32_t gamma  = mod_pow_u(g, (Q-1)/(2*n));
    uint32_t ginv   = mod_pow_u(gamma, Q-2);

    for(int m=2; m<=n; m<<=1){
        int j=m>>1;
        uint32_t gp = mod_pow_u(ginv, n/m);
        for(int k=0;k<n;k+=m){
            for(int j1=0;j1<j;++j1){
                uint32_t tw = mod_pow_u(gp, (uint64_t)(2*j1+1));
                uint32_t u  = a[k+j1];
                uint32_t v  = (uint32_t)(((unsigned long long)a[k+j1+j]*tw)%Q);
                a[k+j1]   = (u+v)%Q;
                a[k+j1+j] = (u+Q-v)%Q;
            }
        }
    }
    // n^{-1}
    uint32_t ninv = mod_pow_u((uint64_t)n, Q-2);
    for(int i=0;i<n;++i) a[i] = (uint32_t)(((unsigned long long)a[i]*ninv)%Q);
}

// Linf (정수 동일성: 0이면 통과)
static uint32_t linf_ntt(const std::vector<uint32_t>& a,
                         const std::vector<uint32_t>& b){
    uint32_t m=0;
    for(size_t i=0;i<a.size();++i){
        uint32_t u=a[i], v=b[i];
        uint32_t d = (u>=v)? (u-v):(v-u);
        if(d>m) m=d;
    }
    return m;
}

int main(){
#if !USE_NTT
    printf("[NTT] Build with -DUSE_NTT=1 to run this test.\n");
    return 0;
#else
    std::vector<uint32_t> a(N);
    for(int i=0;i<N;++i) a[i] = (uint32_t)i;

    elem_t *d_A=nullptr;
    cudaMalloc(&d_A, sizeof(elem_t)*N);
    cudaMemcpy(d_A, a.data(), sizeof(elem_t)*N, cudaMemcpyHostToDevice);

    precompute_and_upload_all();
    int *d_plan=nullptr, *d_offsets=nullptr; make_plan_and_offsets(d_plan, d_offsets);
    size_t shmem = sizeof(elem_t)*N;

    cudaEvent_t evs, eve; cudaEventCreate(&evs); cudaEventCreate(&eve);
    cudaEventRecord(evs);
    launch_forward_bfu(d_A, d_plan, d_offsets, shmem, 1);
    cudaEventRecord(eve); cudaEventSynchronize(eve);
    float ms=0.f; cudaEventElapsedTime(&ms, evs, eve);

    std::vector<uint32_t> A_gpu(N);
    cudaMemcpy(A_gpu.data(), d_A, sizeof(elem_t)*N, cudaMemcpyDeviceToHost);

    printf("NTT(N=%d) elapsed: %.3f ms\n", N, ms);
    printf("NTT(N=%d) of [0..%d]:\n", N, N-1);
    for(int i=0;i<N;++i){ printf("%u\n", A_gpu[i]); }

    cudaEventDestroy(evs); cudaEventDestroy(eve);
    cudaFree(d_A); cudaFree(d_plan); cudaFree(d_offsets);
    return 0;
#endif
}
