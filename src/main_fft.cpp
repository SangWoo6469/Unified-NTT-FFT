#include <vector>
#include <random>
#include <cstdio>
#include <complex>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "config.hpp"
#include "types.hpp"
#include "reference_cpu.hpp"

// extern (kernels & precompute)
void precompute_and_upload_all();
void make_plan_and_offsets(int*& d_plan, int*& d_offsets);
void launch_forward_bfu(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch);

static double linf_fft(const std::vector<std::complex<double>>& a,
                       const std::vector<std::complex<double>>& b){
    double m=0.0; for(size_t i=0;i<a.size();++i){ double e=std::max(std::abs(a[i].real()-b[i].real()), std::abs(a[i].imag()-b[i].imag())); if(e>m) m=e; } return m;
}

static double parseval_rel_err(const std::vector<std::complex<double>>& x,
                               const std::vector<std::complex<double>>& X){
    // Parseval: sum|x|^2 == (1/N) sum|X|^2
    long long n = (long long)x.size();
    long double se_time=0.0L, se_freq=0.0L;
    for(size_t i=0;i<x.size();++i){
        se_time += (long double)std::norm(x[i]);
        se_freq += (long double)std::norm(X[i]);
    }
    long double lhs = se_time;
    long double rhs = se_freq / (long double)n;
    long double rel = std::abs((lhs - rhs) / (std::max((long double)1.0, std::abs(lhs))));
    return (double)rel;
}

int main(){
#if USE_NTT
    printf("[FFT] Build with -DUSE_NTT=0 to run this test.\n");
    return 0;
#else
    std::mt19937 rng(1234); std::uniform_real_distribution<double> U(-1,1);
    std::vector<elem_t> h_in(N); for(int i=0;i<N;++i) h_in[i]={U(rng),U(rng)};

    elem_t* d_buf=nullptr; cudaMalloc(&d_buf, sizeof(elem_t)*N);
    cudaMemcpy(d_buf, h_in.data(), sizeof(elem_t)*N, cudaMemcpyHostToDevice);

    precompute_and_upload_all();
    int *d_plan=nullptr, *d_offsets=nullptr; make_plan_and_offsets(d_plan, d_offsets);
    size_t shmem = sizeof(elem_t)*N;

    cudaEvent_t evs, eve; cudaEventCreate(&evs); cudaEventCreate(&eve);
    cudaEventRecord(evs);
    launch_forward_bfu(d_buf, d_plan, d_offsets, shmem, /*batch=*/1);
    cudaEventRecord(eve); cudaEventSynchronize(eve);
    float ms=0.f; cudaEventElapsedTime(&ms, evs, eve);

    std::vector<elem_t> h_spec(N); cudaMemcpy(h_spec.data(), d_buf, sizeof(elem_t)*N, cudaMemcpyDeviceToHost);

    printf("FFT(N=%d) forward elapsed: %.3f ms\n", N, ms);

    cudaEventDestroy(evs); cudaEventDestroy(eve);
    cudaFree(d_buf); cudaFree(d_plan); cudaFree(d_offsets);
    return 0;
#endif
}
