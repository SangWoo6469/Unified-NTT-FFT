#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "config.hpp"
#include "types.hpp"
#include "device_tables.hpp"

#if USE_DILITHIUM_ZETAS
#include "zetas_dilithium.h"
#endif

#if USE_NTT
extern __constant__ uint32_t d_bfu_m4[1];
extern __constant__ uint32_t d_bfu_m8[8];
extern __constant__ uint32_t d_bfu_m16[18];
extern __constant__ uint32_t d_stage_twiddles_ntt[MAX_STAGES * N];
extern __constant__ uint32_t d_stage_twiddles_ntt_inv[MAX_STAGES * N];
#else
extern __constant__ double2  d_bfu_m4_c[1];
extern __constant__ double2  d_bfu_m8_c[8];
extern __constant__ double2  d_bfu_m16_c[18];
extern __constant__ double2  d_stage_twiddles_fft[MAX_STAGES * N];
extern __constant__ double2  d_stage_twiddles_fft_inv[MAX_STAGES * N];
#endif

static inline uint32_t mod_pow_host(uint64_t a, uint64_t e){
    uint64_t r=1%Q; a%=Q;
    while(e){ if(e&1) r=(r*a)%Q; a=(a*a)%Q; e>>=1; }
    return (uint32_t)r;
}
static inline int bitrev_local(int x, int log2R){
    int r=0; for(int k=0;k<log2R;++k){ r=(r<<1)|(x&1); x>>=1; } return r;
}

// ---------------- Hardcoded FLAT_TW for N=256, PLAN=16,4,4 (drop-1) ----------------
#if USE_NTT && (N==256)
namespace flat_tw_16_4_4 {
static const uint32_t S0[] = {
    4808194, 3765607, 3761513, 5178923, 5496691, 5234739, 5178987, 7778734, 3542485, 2682288, 2129892, 3764867, 7375178, 557458, 7159240
};
static const uint32_t S1[] = {
    5010068, 3602218, 3415069, 5152541, 3415069, 5269599, 4855975, 394148, 2917338, 7737789, 5483103, 1095468, 2663378, 5269599, 6663429, 2453983, 3756790, 3182878, 676590, 2071829, 3192354, 2815639, 2917338, 4361428, 4317364, 2740543, 4805951, 1714295, 7562881, 1159875, 7946292, 1095468, 1858416, 4795319, 556856, 7986269, 6705802, 3704823, 4623627, 1460718, 6663429, 5639874, 7044481, 3241972, 7823561, 2283733, 3345963, 5138445
};
static const uint32_t S2[] = {
    3073009, 1753, 6757063, 5801164, 6757063, 8352605, 1528066, 4234153, 6195333, 1356448, 3994671, 3747250, 2508980, 8352605, 3980599, 7062739, 5346675, 2660408, 4213992, 2998219, 3363542, 348812, 6195333, 7025525, 4183372, 1182243, 507927, 3482206, 6903432, 4829411, 7814814, 3747250, 1900052, 4912752, 3105558, 2192938, 2811291, 6695264, 2647994, 3901472, 3980599, 636927, 1736313, 7921677, 508145, 4561790, 8077412, 4040196, 5744944, 2660408, 527981, 5989328, 586241, 4423473, 1148858, 2387513, 2926054, 2683270, 3363542, 1254190, 1937570, 6400920, 1163598, 3035980, 5767564, 6444997, 1987814, 7025525, 8368538, 1011223, 5365997, 141835, 5258977, 4423672, 1994046, 7080401, 507927, 4969849, 2462444, 3195676, 3531229, 7727142, 860144, 4768667, 7270901, 4829411, 4148469, 2925816, 1665318, 6084020, 5130263, 2039144, 3430436, 6143691, 1900052, 6500539, 1277625, 5720892, 6346610, 1787943, 6006015, 5926272, 482649, 2192938, 6187330, 5604662, 7009900, 4541938, 2028118, 1009365, 1723229, 2461387, 2647994, 2772600, 4892034, 8291116, 7609976, 8052569, 2358373, 6161950, 5157610, 636927, 6545891, 4197502, 2491325, 5925040, 169688, 1239911, 7648983, 2312838, 508145, 4146264, 5396636, 2678278, 3033742, 7153756, 6764887, 7198174, 235407, 4040196, 5274859, 6458164, 4405932, 458740, 3852015, 8321269, 7794176, 6125690, 527981, 1979497, 5418153, 8111961, 3014420, 5601629, 545376, 5184741, 4564692, 4423473, 6715099, 621164, 749577, 2659525, 5183169, 7070156, 1370517, 6026202, 2926054, 89301, 8106357, 5095502, 5889092, 6018354, 7655613, 5702139, 5046034, 1254190, 3974485, 7921254, 140244, 268456, 4158088, 8129971, 7630840, 3374250, 1163598, 3284915, 3258457, 7561656, 8240173, 1744507, 1054478, 818761
};

inline const uint32_t* stage_ptr(int s){
    switch(s){
        case 0: return S0;
        case 1: return S1;
        case 2: return S2;
        default: return nullptr;
    }
}
inline size_t stage_len(int s){
    switch(s){
        case 0: return sizeof(S0)/sizeof(S0[0]);
        case 1: return sizeof(S1)/sizeof(S1[0]);
        case 2: return sizeof(S2)/sizeof(S2[0]);
        default: return 0;
    }
}
} // namespace flat_tw_16_4_4
#endif

// ---------------- BFU m[] 업로드 ----------------
static void upload_bfu_tables(){
#if USE_NTT
    static const uint32_t M4[1]   = { 4808194u };
    static const uint32_t M8[8]   = { 1u, 1u, 1u, 4808194u, 1u, 4808194u, 3763560u, 2047u };
    static const uint32_t M16[18] = { 5178955u, 8380385u, 3014702u, 8249441u, 8193657u, 8249409u, 4808194u, 4808194u, 4616857u, 8378370u, 4616857u, 8378370u, 4808194u, 0u, 0u, 0u, 0u, 0u };
    cudaMemcpyToSymbol(d_bfu_m4,  M4,  sizeof(M4));
    cudaMemcpyToSymbol(d_bfu_m8,  M8,  sizeof(M8));
    cudaMemcpyToSymbol(d_bfu_m16, M16, sizeof(M16));
#else
    // FFT BFU m tables already set below per-user; no changes here
    static const double2 M4[1]  = { { 6.123233995736766e-17, -1.0 } };
    static const double2 M8[8]  = {
        { 1.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 }, { 0.0, -1.0000000000000002 },
        { 1.0, 0.0 }, { 0.0, -1.0000000000000002 }, { -5.5511151231257827e-17, -0.70710678118654768 }, { 0.70710678118654768, 5.5511151231257827e-17 }
    };
    static const double2 M16[18] = {
        { -1.6653345369377348e-16, -0.92387953251128674 },
        { -0.38268343236508984, 5.5511151231257827e-17 },
        { -5.5511151231257827e-17, -0.38268343236508962 },
        { 0.92387953251128674, -1.6653345369377348e-16 },
        { -2.2204460492503131e-16, -1.3065629648763764 },
        { 0.5411961001461969, -1.1102230246251565e-16 },
        { -2.2204460492503131e-16, -1.0 },
        { -2.2204460492503131e-16, -1.0 },
        { -1.1102230246251565e-16, -0.70710678118654746 },
        { 0.70710678118654757, -1.1102230246251565e-16 },
        { -1.1102230246251565e-16, -0.70710678118654746 },
        { 0.70710678118654757, -1.1102230246251565e-16 },
        { -2.2204460492503131e-16, -1.0 },
        { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }
    };
    cudaMemcpyToSymbol(d_bfu_m4_c,  M4,  sizeof(M4));
    cudaMemcpyToSymbol(d_bfu_m8_c,  M8,  sizeof(M8));
    cudaMemcpyToSymbol(d_bfu_m16_c, M16, sizeof(M16));
#endif
}

#if !USE_NTT
// Build flat drop-1 FFT twiddles per provided plan
static void build_flat_twiddles_fft_drop1(std::vector<double2>& flat0,
                                          std::vector<double2>& flat1,
                                          std::vector<double2>& flat2){
    auto w = [&](int R){ double t=-2.0*M_PI/R; return make_double2(cos(t), sin(t)); };
    auto mulc=[&](double2 a,double2 b){ return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); };
    auto powc=[&](double2 ww,int e){ double2 r=make_double2(1,0); for(int i=0;i<e;++i) r=mulc(r,ww); return r; };
    int j = 1;
    for(int s=0;s<MAX_STAGES;++s){
        int R = PLAN[s];
        int cols = j;
        int rm1 = R-1;
        std::vector<double2>* tgt = (s==0? &flat0 : (s==1? &flat1 : &flat2));
        tgt->resize(cols*rm1);
        double2 ww = w(R);
        for(int j1=0;j1<cols;++j1){
            for(int i=1;i<R;++i){
                // odd-exponent schedule: exponent = (2*j1+1) * i
                int e = (2*j1+1) * i;
                (*tgt)[ j1*rm1 + (i-1) ] = powc(ww, e % (R));
            }
        }
        j *= R;
    }
}
#endif

// ---------------- Stage twiddle (bit-rev 흡수) ----------------
#if USE_NTT
static void build_stage_twiddles_ntt(std::vector<uint32_t>& TW, std::vector<uint32_t>& TWi){
#if N==256
    (void)TWi;
    TW.clear(); TW.reserve(sizeof(flat_tw_16_4_4::S0)/4 + sizeof(flat_tw_16_4_4::S1)/4 + sizeof(flat_tw_16_4_4::S2)/4);
    TW.insert(TW.end(), flat_tw_16_4_4::S0, flat_tw_16_4_4::S0 + sizeof(flat_tw_16_4_4::S0)/sizeof(uint32_t));
    TW.insert(TW.end(), flat_tw_16_4_4::S1, flat_tw_16_4_4::S1 + sizeof(flat_tw_16_4_4::S1)/sizeof(uint32_t));
    TW.insert(TW.end(), flat_tw_16_4_4::S2, flat_tw_16_4_4::S2 + sizeof(flat_tw_16_4_4::S2)/sizeof(uint32_t));
    cudaMemcpyToSymbol(d_stage_twiddles_ntt, TW.data(), TW.size()*sizeof(uint32_t));
#else
    TW.assign(MAX_STAGES*N, 0u);
    TWi.assign(MAX_STAGES*N, 0u);
    uint32_t g=3;
    uint32_t gamma = mod_pow_host(g, (Q-1)/(2*N));
    uint32_t gamma_inv = mod_pow_host(gamma, Q-2);
    int stride=1, woff=0;
    for(int s=0;s<MAX_STAGES;++s){
        int R=PLAN[s];
        int log2R = (R==1?0:(int)std::round(std::log2(R)));
        uint32_t gp  = mod_pow_host(gamma,     N/(R*stride));
        uint32_t gpi = mod_pow_host(gamma_inv, N/(R*stride));
        for(int i=0;i<R;++i){
            int ib = bitrev_local(i, log2R);
            for(int j1=0;j1<stride;++j1){
                uint64_t e = (uint64_t)ib * (uint64_t)(2*j1+1);
                TW [woff + (i*stride + j1)] = mod_pow_host(gp , e);
                TWi[woff + (i*stride + j1)] = mod_pow_host(gpi, e);
            }
        }
        woff += (R*stride);
        stride *= R;
    }
    cudaMemcpyToSymbol(d_stage_twiddles_ntt,     TW .data(), sizeof(uint32_t)*TW .size());
    cudaMemcpyToSymbol(d_stage_twiddles_ntt_inv, TWi.data(), sizeof(uint32_t)*TWi.size());
#endif
}
#else
static void build_stage_twiddles_fft(std::vector<double2>& TW, std::vector<double2>& TWi){
#if (N==512) || (N==1024)
    // Build flat drop-1 twiddles per provided plan (identical to given tables)
    std::vector<double2> S0, S1, S2;
    build_flat_twiddles_fft_drop1(S0, S1, S2);
    // Upload S0||S1||S2 contiguously into d_stage_twiddles_fft
    TW.clear();
    TW.insert(TW.end(), S0.begin(), S0.end());
    TW.insert(TW.end(), S1.begin(), S1.end());
    TW.insert(TW.end(), S2.begin(), S2.end());
    cudaMemcpyToSymbol(d_stage_twiddles_fft, TW.data(), TW.size()*sizeof(double2));
    // No inverse flat needed here (not used in current FFT path)
    TWi.clear();
#else
    TW .assign(MAX_STAGES*N, make_double2(0,0));
    TWi.assign(MAX_STAGES*N, make_double2(0,0));
    int stride=1, woff=0;
    for(int s=0;s<MAX_STAGES;++s){
        int R=PLAN[s];
        int log2R = (R==1?0:(int)std::round(std::log2(R)));
        for(int i=0;i<R;++i){
            int ib = bitrev_local(i, log2R);
            for(int j1=0;j1<stride;++j1){
                double theta_fwd = (double)(N/(R*stride)) * (double)(ib*(2*j1+1)) * (-M_PI/(double)N);
                double theta_inv = -theta_fwd;
                TW [woff + (i*stride + j1)] = make_double2(cos(theta_fwd), sin(theta_fwd));
                TWi[woff + (i*stride + j1)] = make_double2(cos(theta_inv), sin(theta_inv));
            }
        }
        woff += (R*stride);
        stride *= R;
    }
    cudaMemcpyToSymbol(d_stage_twiddles_fft,     TW .data(), sizeof(double2)*TW .size());
    cudaMemcpyToSymbol(d_stage_twiddles_fft_inv, TWi.data(), sizeof(double2)*TWi.size());
#endif
}
#endif

// ---- 외부 노출 함수 ----
void precompute_and_upload_all(){
    upload_bfu_tables();
#if USE_NTT
    std::vector<uint32_t> TW, TWi;
    build_stage_twiddles_ntt(TW, TWi);
#else
    std::vector<double2> TW, TWi;
    build_stage_twiddles_fft(TW, TWi);
#endif
}

void make_plan_and_offsets(int*& d_plan, int*& d_offsets){
    int h_plan[MAX_STAGES]; int h_off[MAX_STAGES];
    int stride=1; int pref=0;
    for(int s=0;s<MAX_STAGES;++s){
        h_plan[s]=PLAN[s];
        h_off[s]=pref;
        pref += PLAN[s]*stride;
        stride*=PLAN[s];
    }
    cudaMalloc(&d_plan, sizeof(int)*MAX_STAGES);
    cudaMalloc(&d_offsets, sizeof(int)*MAX_STAGES);
    cudaMemcpy(d_plan, h_plan, sizeof(int)*MAX_STAGES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_off,  sizeof(int)*MAX_STAGES, cudaMemcpyHostToDevice);
}

// 런처 선언 (kernels.cu)
void launch_forward_bfu(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch);
void launch_ifft(elem_t* d_buf, const int* d_plan, const int* d_offsets, size_t shmem, int batch);
void launch_intt(elem_t* d_buf, size_t shmem, int batch);
