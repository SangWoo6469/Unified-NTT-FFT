#include <cuda_runtime.h>
#include "config.hpp"
#include "types.hpp"
#include "bfu.hpp"
#include "device_tables.hpp"

// ====== __constant__ 정의 ======
#if USE_NTT
__constant__ uint32_t d_bfu_m4[1];
__constant__ uint32_t d_bfu_m8[8];
__constant__ uint32_t d_bfu_m16[18];
__constant__ uint32_t d_stage_twiddles_ntt[MAX_STAGES * N];
__constant__ uint32_t d_stage_twiddles_ntt_inv[MAX_STAGES * N];
#else
__constant__ double2  d_bfu_m4_c[1];
__constant__ double2  d_bfu_m8_c[8];
__constant__ double2  d_bfu_m16_c[18];
__constant__ double2  d_stage_twiddles_fft[MAX_STAGES * N];
__constant__ double2  d_stage_twiddles_fft_inv[MAX_STAGES * N];
#endif

// ====== 정방향 BFU 퓨전 ======
template<int STAGES>
__global__ void kernel_forward_bfu_fused(elem_t* __restrict__ d_data,
                                         const int* __restrict__ d_plan,
                                         const int* __restrict__ d_offsets){
    extern __shared__ elem_t smem[];
    const int poly = blockIdx.x;
    elem_t* data = d_data + poly * N;

    // load to shared
    for(int i=threadIdx.x; i<N; i+=blockDim.x) smem[i]=data[i];
    __syncthreads();

#if USE_NTT && (N==256)
    // bit-reverse permute input once
    auto bitrev=[&](int x)->int{ int b=8; // log2(256)
        int r=0; for(int k=0;k<b;++k){ r=(r<<1)|(x&1); x>>=1; } return r; };
    for(int i=threadIdx.x;i<N;i+=blockDim.x){ int j=bitrev(i); if(j>i){ uint32_t t=smem[i]; smem[i]=smem[j]; smem[j]=t; } }
    __syncthreads();
#endif

    int stride = 1;
    for(int s=0; s<STAGES; ++s){
        const int R = d_plan[s];
        const int groups = N / (R * stride);
        const int base   = d_offsets[s];

        for(int g = threadIdx.x; g < groups*stride; g += blockDim.x){
            int j1 = g % stride;
            int gid= g / stride;

            elem_t lane[16];

            #pragma unroll
            for(int i=0;i<R;++i){
                int idx = gid*R*stride + j1 + i*stride;
#if USE_NTT && (N==256)
                int flat_base = (s==0?0:(s==1?15:(15+48)));
                if(i==0){
                    lane[i] = smem[idx];
                } else {
                    int tw_idx = flat_base + j1*(R-1) + (i-1);
                    uint32_t tw = d_stage_twiddles_ntt[ tw_idx ];
                    lane[i] = mul_mod(smem[idx], tw);
                }
#else
#if USE_NTT
                uint32_t tw = d_stage_twiddles_ntt[ base + (i*stride + j1) ];
                lane[i] = mul_mod(smem[idx], tw);
#else
                double2 tw = d_stage_twiddles_fft[ base + (i*stride + j1) ];
                lane[i] = cmul(smem[idx], {tw.x, tw.y});
#endif
#endif
            }

            switch(R){
                case 2:  BFU_radix2(lane);  break;
                case 4:  BFU_radix4(lane);  break;
                case 8:  BFU_radix8(lane);  break;
                case 16: BFU_radix16(lane); break;
                default: break;
            }

            #pragma unroll
            for(int i=0;i<R;++i){
                int idx = gid*R*stride + j1 + i*stride;
                smem[idx] = lane[i];
            }
        }
        __syncthreads();
        stride *= R;
    }

    for(int i=threadIdx.x; i<N; i+=blockDim.x) data[i]=smem[i];
}

// ====== 역변환(radix-2) : FFT(iFFT) / NTT(INTT) ======
__global__ void kernel_ifft_radix2(elem_t* __restrict__ d_data,
                                   const int* __restrict__ d_plan,
                                   const int* __restrict__ d_offsets){
#if USE_NTT
    // FFT 전용
#else
    extern __shared__ elem_t smem[];
    const int poly = blockIdx.x;
    elem_t* data = d_data + poly*N;

    for(int i=threadIdx.x;i<N;i+=blockDim.x) smem[i]=data[i];
    __syncthreads();

    auto bitrev=[&](int x)->int{ int b=0; while((1<<b)<N) ++b; int r=0; for(int k=0;k<b;++k){ r=(r<<1)|(x&1); x>>=1; } return r; };
    for(int i=threadIdx.x;i<N;i+=blockDim.x){ int j=bitrev(i); if(j>i){ elem_t t=smem[i]; smem[i]=smem[j]; smem[j]=t; } }
    __syncthreads();

    for(int m=2; m<=N; m<<=1){
        int j = m>>1;
        for(int k=threadIdx.x; k<N; k+=blockDim.x){
            int blk=(k/m)*m, j1=k-blk; if(j1>=j) continue;
            int i0=blk+j1, i1=i0+j;
            double theta = (double)(N/m) * (double)(2*j1+1) * (+M_PI/(double)N);
            cplx tw = {cos(theta), sin(theta)};
            cplx t  = cmul(smem[i1], tw);
            cplx u  = smem[i0];
            smem[i0]= cadd(u,t);
            smem[i1]= csub(u,t);
        }
        __syncthreads();
    }
    double invN=1.0/(double)N;
    for(int i=threadIdx.x;i<N;i+=blockDim.x){ smem[i].x*=invN; smem[i].y*=invN; }
    for(int i=threadIdx.x;i<N;i+=blockDim.x) data[i]=smem[i];
#endif
}

__global__ void kernel_intt_radix2(elem_t* __restrict__ d_data){
#if USE_NTT
    extern __shared__ elem_t smem[];
    const int poly = blockIdx.x;
    elem_t* data = d_data + poly*N;

    for(int i=threadIdx.x;i<N;i+=blockDim.x) smem[i]=data[i];
    __syncthreads();

    auto bitrev=[&](int x)->int{ int b=0; while((1<<b)<N) ++b; int r=0; for(int k=0;k<b;++k){ r=(r<<1)|(x&1); x>>=1; } return r; };
    for(int i=threadIdx.x;i<N;i+=blockDim.x){ int j=bitrev(i); if(j>i){ uint32_t t=smem[i]; smem[i]=smem[j]; smem[j]=t; } }
    __syncthreads();

    for(int m=2; m<=N; m<<=1){
        int j = m>>1;
        for(int k=threadIdx.x; k<N; k+=blockDim.x){
            int blk=(k/m)*m, j1=k-blk; if(j1>=j) continue;
            int i0=blk+j1, i1=i0+j;
            uint32_t t = smem[i1];
            uint32_t u = smem[i0];
            smem[i0]= add_mod(u,t);
            smem[i1]= sub_mod(u,t);
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<N;i+=blockDim.x) data[i]=smem[i];
#else
    // FFT가 아니므로 no-op
#endif
}

// ---- 런처 ----
void launch_forward_bfu(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch){
    dim3 grid(batch), block(32);
    kernel_forward_bfu_fused<MAX_STAGES><<<grid, block, shmem>>>(d_buf, d_plan, d_offsets);
    cudaDeviceSynchronize();
}
void launch_ifft(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch){
    dim3 grid(batch), block(32);
    kernel_ifft_radix2<<<grid, block, shmem>>>(d_buf, d_plan, d_offsets);
    cudaDeviceSynchronize();
}
void launch_intt(elem_t* d_buf, size_t shmem, int batch){
    dim3 grid(batch), block(32);
    kernel_intt_radix2<<<grid, block, shmem>>>(d_buf);
    cudaDeviceSynchronize();
}
