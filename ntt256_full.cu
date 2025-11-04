// ntt256_bench_all.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

// ================= Global params =================
static constexpr int      N     = 256;
static constexpr uint32_t Q     = 8380417u; // modulus
static constexpr uint32_t GAMMA = 1753u;    // drop-1 base

// === BFU 내부 상수(네가 준 값) ===
static constexpr uint32_t OMEGA4  = 4808194u; // Radix-4 내부 곱
static constexpr uint32_t OMEGA8  = 3765607u; // Radix-8 m[] 생성용
static constexpr uint32_t OMEGA16 = 5178923u; // Radix-16 m[] 생성용

// ================= Device constants (Barrett) =================
__constant__ uint32_t d_Q;     // modulus
__constant__ uint64_t d_mu;    // floor(2^64 / Q)

// ================= Device helpers =================
// ----- Lazy add/sub (no conditional reduction) -----
__device__ __forceinline__ uint32_t add_lazy(uint32_t a, uint32_t b){
    // 그냥 더함 (감축 없음)
    return a + b;
}
__device__ __forceinline__ uint32_t sub_lazy(uint32_t a, uint32_t b){
    // 언더플로우 방지만 수행 (감축 없음)
    return a -b;
}

// ----- Barrett mul: r = a*b mod Q (k = 64) -----
// d_mu = floor(2^64 / Q)
__device__ __forceinline__ uint32_t mul_barrett(uint32_t a, uint32_t b){
    uint64_t x  = (uint64_t)a * (uint64_t)b;      // 64-bit product
    uint64_t t  = __umul64hi(x, d_mu);            // floor((x*2^64)/Q)
    uint64_t r  = x - t * (uint64_t)d_Q;          // tentative remainder
    if (r >= d_Q) r -= d_Q;
    if (r >= d_Q) r -= d_Q;
    return (uint32_t)r;
}

__device__ __forceinline__ unsigned bitrev8_dev(unsigned x){
    x = ((x & 0xF0u) >> 4) | ((x & 0x0Fu) << 4);
    x = ((x & 0xCCu) >> 2) | ((x & 0x33u) << 2);
    x = ((x & 0xAAu) >> 1) | ((x & 0x55u) << 1);
    return x;
}

// ================= Shared buffers =================
__shared__ uint32_t svec[N];
__shared__ uint32_t s0_accum[128]; // R=16 중간합
__shared__ uint32_t s1_accum[128];

// ================= Stage descriptor =================
struct StageDesc {
    int R;                // 2/4/8/16
    int j;                // 누적 stride
    const uint32_t* flat; // drop-1 table, len = j*(R-1)
};

// ================= BFU consts =================
__constant__ uint32_t d_m4_omega;  // Radix-4 내부 곱 계수
__constant__ uint32_t d_m8[8];     // Radix-8 내부 곱 계수
__constant__ uint32_t d_m16[13];   // Radix-16 내부 곱 계수

// ================= Addr helper =================
__device__ __forceinline__ int addr_loc(const StageDesc& stg, int bfu, int i){
    int R = stg.R, j = stg.j;
    int k = bfu / j, j1 = bfu % j;
    int base = R * k * j;
    return base + j1 + i * j;
}

// ================= drop-1 pre-twiddle =================
__device__ __forceinline__
void drop1_pretwiddle_parallel(const StageDesc& stg, int bfu, int lane){
    int R = stg.R, j = stg.j;
    int threadsPerBFU = R >> 1;
    int iA = lane + 1;
    int iB = lane + 1 + threadsPerBFU;

    int k = bfu / j, j1 = bfu % j;
    int base = R * k * j;
    int off  = j1 * (R - 1);

    if (iA < R){
        int idx = base + j1 + iA * j;
        uint32_t w = stg.flat[off + (iA - 1)];
        svec[idx] = mul_barrett(svec[idx], w);
    }
    if (iB < R){
        int idx = base + j1 + iB * j;
        uint32_t w = stg.flat[off + (iB - 1)];
        svec[idx] = mul_barrett(svec[idx], w);
    }
}

// ================= Radix-2 =================
template<int R>
__device__ __forceinline__
void r2_step(const StageDesc& stg, int bfu){
    int j = stg.j;
    int k = bfu / j, j1 = bfu % j, base = R * k * j;
    int ia = base + j1 + 0 * j;
    int ib = base + j1 + 1 * j;
    uint32_t a = svec[ia], b = svec[ib];
    svec[ia] = add_lazy(a,b);
    svec[ib] = sub_lazy(a,b);
}

// ================= Radix-4 (네 정의) =================
// BFU당 스레드 = 2, lane ∈ {0,1}
__device__ __forceinline__
void r4_step1_pairs(const StageDesc& stg, int bfu, int lane){
    int a = (lane == 0) ? 0 : 2;
    int b = a + 1;
    int ia = addr_loc(stg,bfu,a), ib = addr_loc(stg,bfu,b);
    uint32_t xa = svec[ia], xb = svec[ib];
    svec[ia] = add_lazy(xa,xb); svec[ib] = sub_lazy(xa,xb);
}
__device__ __forceinline__
void r4_step2_twiddle(const StageDesc& stg, int bfu, int lane){
    if (lane == 1){
        int i3 = addr_loc(stg,bfu,3);
        svec[i3] = mul_barrett(svec[i3], d_m4_omega);
    }
}
__device__ __forceinline__
void r4_step3_pairs(const StageDesc& stg, int bfu, int lane){
    int a = (lane == 0) ? 0 : 1;
    int b = a + 2;
    int ia = addr_loc(stg,bfu,a), ib = addr_loc(stg,bfu,b);
    uint32_t xa = svec[ia], xb = svec[ib];
    svec[ia] = add_lazy(xa,xb); svec[ib] = sub_lazy(xa,xb);
}

// ================= Radix-8 (네 정의) =================
// BFU당 스레드 = 4, lane ∈ {0,1,2,3}
__device__ __forceinline__
void r8_step1_pairs(const StageDesc& stg, int bfu, int lane){
    const int P[4][2] = { {0,1},{2,3},{4,5},{6,7} };
    int a = P[lane][0], b = P[lane][1];
    int ia = addr_loc(stg,bfu,a), ib = addr_loc(stg,bfu,b);
    uint32_t xa = svec[ia], xb = svec[ib];
    svec[ia] = add_lazy(xa,xb); svec[ib] = sub_lazy(xa,xb);
}
__device__ __forceinline__
void r8_step2_mid(const StageDesc& stg, int bfu, int lane){
    if (lane == 2){
        int ia=addr_loc(stg,bfu,4), ib=addr_loc(stg,bfu,6);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    } else if (lane == 3){
        int ia=addr_loc(stg,bfu,5), ib=addr_loc(stg,bfu,7);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
}
__device__ __forceinline__
void r8_step3_twiddle(const StageDesc& stg, int bfu, int lane){
    if (lane == 1){ int i=addr_loc(stg,bfu,3); svec[i] = mul_barrett(svec[i], d_m8[3]); }
    if (lane == 3){ int i=addr_loc(stg,bfu,6); svec[i] = mul_barrett(svec[i], d_m8[5]); }
    if (lane == 2){ int i=addr_loc(stg,bfu,5); svec[i] = mul_barrett(svec[i], d_m8[6]); }
    if (lane == 0){ int i=addr_loc(stg,bfu,7); svec[i] = mul_barrett(svec[i], d_m8[7]); }
}
__device__ __forceinline__
void r8_step4_pairs(const StageDesc& stg, int bfu, int lane){
    if (lane == 0){
        int ia=addr_loc(stg,bfu,0), ib=addr_loc(stg,bfu,2);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    } else if (lane == 1){
        int ia=addr_loc(stg,bfu,1), ib=addr_loc(stg,bfu,3);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    } else if (lane == 2){
        int ia=addr_loc(stg,bfu,5), ib=addr_loc(stg,bfu,7);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
}
__device__ __forceinline__
void r8_step5_final(const StageDesc& stg, int bfu, int lane){
    const int P[4][2] = { {0,4},{1,5},{2,6},{3,7} };
    int a = P[lane][0], b = P[lane][1];
    int ia = addr_loc(stg,bfu,a), ib = addr_loc(stg,bfu,b);
    uint32_t xa = svec[ia], xb = svec[ib];
    svec[ia] = add_lazy(xa,xb); svec[ib] = sub_lazy(xa,xb);
}

// ================= Radix-16 (네 정의 + Output 페어 수정) =================
// BFU당 스레드 = 8, lane ∈ {0..7}
__device__ __forceinline__
void r16_p1(const StageDesc& stg, int bfu, int lane){
    int a = 2*lane, b = a+1;
    int ia = addr_loc(stg,bfu,a), ib = addr_loc(stg,bfu,b);
    uint32_t xa=svec[ia], xb=svec[ib];
    svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
}
__device__ __forceinline__
void r16_p2(const StageDesc& stg, int bfu, int lane){
    const int P[7][2]={{0,2},{4,6},{5,7},{8,10},{12,14},{11,13},{9,15}};
    if (lane<7){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg,bfu,a), ib=addr_loc(stg,bfu,b);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
}
__device__ __forceinline__
void r16_p3(const StageDesc& stg, int bfu, int lane){
    if (lane==0){
        uint32_t v9 = svec[addr_loc(stg,bfu,9)];
        uint32_t v11= svec[addr_loc(stg,bfu,11)];
        uint32_t v13= svec[addr_loc(stg,bfu,13)];
        uint32_t v15= svec[addr_loc(stg,bfu,15)];
        s0_accum[bfu]=add_lazy(v9,v11);
        s1_accum[bfu]=add_lazy(v13,v15);
    }
    const int P[3][2]={{10,14},{8,12},{0,4}};
    if (lane<3){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg,bfu,a), ib=addr_loc(stg,bfu,b);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
}
__device__ __forceinline__
void r16_p4_tw(const StageDesc& stg, int bfu, int lane){
    switch(lane){
      case 0:{ uint32_t s0=s0_accum[bfu], s1=s1_accum[bfu];
               s0_accum[bfu]=mul_barrett(s0,d_m16[4]); s1_accum[bfu]=mul_barrett(s1,d_m16[5]); } break;
      case 1:{ int i=addr_loc(stg,bfu,9 ); svec[i]=mul_barrett(svec[i], d_m16[0]); } break;
      case 2:{ int i=addr_loc(stg,bfu,15); svec[i]=mul_barrett(svec[i], d_m16[1]); } break;
      case 3:{ int i=addr_loc(stg,bfu,11); svec[i]=mul_barrett(svec[i], d_m16[2]); } break;
      case 4:{ int i=addr_loc(stg,bfu,13); svec[i]=mul_barrett(svec[i], d_m16[3]); } break;
      case 5:{ int i=addr_loc(stg,bfu,3 ); svec[i]=mul_barrett(svec[i], d_m16[6]); } break;
      case 6:{ int i=addr_loc(stg,bfu,12); svec[i]=mul_barrett(svec[i], d_m16[7]); } break;
      case 7:{ int i=addr_loc(stg,bfu,10); svec[i]=mul_barrett(svec[i], d_m16[8]);
               i=addr_loc(stg,bfu,14);     svec[i]=mul_barrett(svec[i], d_m16[9]); } break;
    }
}
__device__ __forceinline__
void r16_p5(const StageDesc& stg, int bfu, int lane){
    if (lane==0){
        uint32_t s0=s0_accum[bfu], s1=s1_accum[bfu];
        s0_accum[bfu]=add_lazy(s0,s1);
        s1_accum[bfu]=sub_lazy(s0,s1);
        int i5=addr_loc(stg,bfu,5);  svec[i5]=mul_barrett(svec[i5], d_m16[10]);
        int i7=addr_loc(stg,bfu,7);  svec[i7]=mul_barrett(svec[i7], d_m16[11]);
        int i6=addr_loc(stg,bfu,6);  svec[i6]=mul_barrett(svec[i6], d_m16[12]);
    }
    const int P[6][2]={{1,3},{2,6},{10,14},{9,15},{11,13},{5,7}};
    if (lane<6){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg,bfu,a), ib=addr_loc(stg,bfu,b);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
}
__device__ __forceinline__
void r16_p6(const StageDesc& stg, int bfu, int lane){
    const int P[4][2]={{1,5},{3,7},{9,11},{15,13}};
    if (lane<4){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg,bfu,a), ib=addr_loc(stg,bfu,b);
        uint32_t xa=svec[ia], xb=svec[ib];
        svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    }
    if (lane==0){
        int i9 =addr_loc(stg,bfu,9 ); svec[i9 ]=sub_lazy(s0_accum[bfu], svec[i9 ]);
        int i15=addr_loc(stg,bfu,15); svec[i15]=sub_lazy(s1_accum[bfu], svec[i15]);
    }
}
__device__ __forceinline__
void r16_p7(const StageDesc& stg, int bfu, int lane){
    const int P[8][2]={{0,8},{1,9},{2,10},{4,12},{6,14},{7,15},{3,13},{5,11}};
    int a=P[lane][0], b=P[lane][1];
    int ia=addr_loc(stg,bfu,a), ib=addr_loc(stg,bfu,b);
    uint32_t xa=svec[ia], xb=svec[ib];
    svec[ia]=add_lazy(xa,xb); svec[ib]=sub_lazy(xa,xb);
    if (lane==7){
        int i11=addr_loc(stg,bfu,11), i13=addr_loc(stg,bfu,13);
        uint32_t t=svec[i11]; svec[i11]=svec[i13]; svec[i13]=t;
    }
}

// ================= Stage executor =================
__device__ void run_stage_like_python(const StageDesc& stg){
    int tid = threadIdx.x;         // 0..127
    int R   = stg.R;
    int tpb = R >> 1;              // R/2
    int bfu = tid / tpb;           // 0..(256/R-1)
    int lane= tid % tpb;           // BFU 내부 스레드
    int BFUs= N / R;
    if (bfu >= BFUs) return;

    // 1) pre-twiddle
    drop1_pretwiddle_parallel(stg, bfu, lane);
    __syncthreads();

    // 2) BFU 내부 스텝들(스텝마다 배리어)
    if (R == 2){
        if (lane==0) r2_step<2>(stg,bfu); __syncthreads();
    } else if (R == 4){
        r4_step1_pairs(stg,bfu,lane);   __syncthreads();
        r4_step2_twiddle(stg,bfu,lane); __syncthreads();
        r4_step3_pairs(stg,bfu,lane);   __syncthreads();
    } else if (R == 8){
        r8_step1_pairs(stg,bfu,lane);   __syncthreads();
        r8_step2_mid(stg,bfu,lane);     __syncthreads();
        r8_step3_twiddle(stg,bfu,lane); __syncthreads();
        r8_step4_pairs(stg,bfu,lane);   __syncthreads();
        r8_step5_final(stg,bfu,lane);   __syncthreads();
    } else { // R == 16
        r16_p1(stg,bfu,lane);    __syncthreads();
        r16_p2(stg,bfu,lane);    __syncthreads();
        r16_p3(stg,bfu,lane);    __syncthreads();
        r16_p4_tw(stg,bfu,lane); __syncthreads();
        r16_p5(stg,bfu,lane);    __syncthreads();
        r16_p6(stg,bfu,lane);    __syncthreads();
        r16_p7(stg,bfu,lane);    __syncthreads();
    }
}

// ================= Kernel =================
__global__ void ntt256_kernel(
    uint32_t* __restrict__ A,
    const StageDesc* __restrict__ stages,
    int numStages)
{
    int tid = threadIdx.x; // 0..127

    // bit-reverse load (2개씩)
    unsigned i0 = tid, i1 = tid + 128;
    svec[i0] = A[bitrev8_dev(i0)];
    svec[i1] = A[bitrev8_dev(i1)];
    __syncthreads();

    for (int s=0; s<numStages; ++s){
        run_stage_like_python(stages[s]);
    }

    A[i0] = svec[i0];
    A[i1] = svec[i1];
}

// ================= Host helpers =================
static inline uint32_t modpowU(uint32_t a, uint64_t e, uint32_t q=Q){
    uint64_t r=1, x=a; while(e){ if(e&1) r=(r*x)%q; x=(x*x)%q; e>>=1; } return (uint32_t)r;
}
static inline unsigned bitrev_host(unsigned x, int bits){
    unsigned r=0; for(int i=0;i<bits;i++){ r=(r<<1)|(x&1); x>>=1; } return r;
}
static std::vector<uint32_t> build_flat_stage(int R, int j){
    int index = N / (R * j);
    uint32_t gp = modpowU(GAMMA, index, Q);
    int rev_bits = 0; while ((1<<rev_bits) < R) ++rev_bits;
    std::vector<uint32_t> out(j * (R - 1));
    for (int j1=0; j1<j; ++j1){
        std::vector<uint32_t> t(R), tb(R);
        for (int i=0;i<R;i++){
            uint64_t e = (uint64_t)i * (2*j1 + 1);
            t[i] = modpowU(gp, e, Q);
        }
        for (int i=0;i<R;i++)
            tb[ bitrev_host((unsigned)i, rev_bits) ] = t[i];
        for (int i=1;i<R;i++)
            out[j1*(R-1) + (i-1)] = tb[i];
    }
    return out;
}
static void build_m4(uint32_t omega4, uint32_t& out){
    out = omega4; // a[3]*=omega4
}
static void build_m8(uint32_t omega, uint32_t q, uint32_t m[8]){
    uint32_t inv2 = (q + 1) / 2;
    for (int i : {0,1,2,4}) m[i] = 1;
    uint32_t w1 = modpowU(omega,1,q);
    uint32_t w2 = modpowU(omega,2,q);
    uint32_t w3 = modpowU(omega,3,q);
    m[3] = w2; m[5] = w2;
    m[6] = (uint32_t)(( (uint64_t)((w1 + w3) % q) * inv2 ) % q);
    { uint32_t t = (w1 + q - w3) % q;
      m[7] = (uint32_t)(( (uint64_t)t * inv2 ) % q); }
}
static void build_m16(uint32_t omega, uint32_t q, uint32_t m[13]){
    uint32_t inv2 = (q + 1) / 2;
    auto pw = [&](int e){ return modpowU(omega,e,q); };
    m[0]  = ( ( (uint64_t)((pw(3) + pw(5)) % q) * inv2 ) % q );
    m[1]  = ( ( (uint64_t)(( (pw(5) + q) - pw(3) ) % q) * inv2 ) % q );
    m[2]  = ( ( (uint64_t)((pw(1) + pw(7)) % q) * inv2 ) % q );
    m[3]  = ( ( (uint64_t)(( (pw(1) + q) - pw(7) ) % q) * inv2 ) % q );
    m[4]  = ( m[2] + m[0] ) % q;
    m[5]  = ( m[1] + m[3] ) % q;
    m[6]  = pw(4);
    m[7]  = pw(4);
    m[12] = pw(4);
    auto mix = [&](int a,int b){ return (uint32_t)(( (uint64_t)((pw(a)+pw(b)) % q) * inv2 ) % q); };
    auto dif = [&](int a,int b){ uint32_t t=( (pw(a)+q) - pw(b) ) % q; return (uint32_t)(( (uint64_t)t * inv2 ) % q); };
    m[8]  = mix(2,6);
    m[10] = mix(2,6);
    m[9]  = dif(2,6);
    m[11] = dif(2,6);
}
static std::string plan_to_string(const std::vector<int>& plan){
    std::string s; for (size_t i=0;i<plan.size();++i){ s += std::to_string(plan[i]); if (i+1<plan.size()) s += "-"; }
    return s;
}

// ===== 모든 (순서 포함) radix 조합 생성: factors ∈ {2,4,8,16}, product=256 =====
static void gen_plans_rec(int prod, std::vector<int>& cur, std::vector<std::vector<int>>& out){
    if (prod == N){ out.push_back(cur); return; }
    for (int f : {2,4,8,16}){
        if (N % (prod * f) == 0){
            cur.push_back(f);
            gen_plans_rec(prod * f, cur, out);
            cur.pop_back();
        }
    }
}
static std::vector<std::vector<int>> generate_all_plans(){
    std::vector<std::vector<int>> out;
    std::vector<int> cur;
    gen_plans_rec(1, cur, out);
    return out;
}

// ================= Runner for one plan =================
struct StagePack {
    std::vector<StageDesc> hstages;
    std::vector<std::vector<uint32_t>> hflats;
    std::vector<uint32_t*> dflats;
    StageDesc* dstages = nullptr;
    int numStages = 0;
};
static StagePack build_stagepack_for_plan(const std::vector<int>& plan){
    StagePack P;
    P.hflats.resize(plan.size());
    P.dflats.resize(plan.size(), nullptr);
    int j = 1;
    for (size_t s=0; s<plan.size(); ++s){
        int R = plan[s];
        P.hflats[s] = build_flat_stage(R, j);

        cudaMalloc(&P.dflats[s], P.hflats[s].size()*sizeof(uint32_t));
        cudaMemcpy(P.dflats[s], P.hflats[s].data(), P.hflats[s].size()*sizeof(uint32_t), cudaMemcpyHostToDevice);

        StageDesc sd; sd.R = R; sd.j = j; sd.flat = P.dflats[s];
        P.hstages.push_back(sd);
        j *= R;
    }
    cudaMalloc(&P.dstages, P.hstages.size()*sizeof(StageDesc));
    cudaMemcpy(P.dstages, P.hstages.data(), P.hstages.size()*sizeof(StageDesc), cudaMemcpyHostToDevice);
    P.numStages = (int)P.hstages.size();
    return P;
}
static void free_stagepack(StagePack& P){
    for (auto p : P.dflats) if (p) cudaFree(p);
    if (P.dstages) cudaFree(P.dstages);
    P = StagePack{};
}

// ================= Main =================
int main(int argc, char** argv){
    int runs = 50;
    unsigned seed = 0x1234;
    if (argc >= 2) runs = std::max(1, atoi(argv[1]));
    if (argc >= 3) seed = (unsigned)strtoul(argv[2], nullptr, 0);

    // 0) Barrett params upload
    {
        uint32_t hQ = Q;                      // 8380417
        // mu = floor(2^64 / Q)
        unsigned __int128 one = (unsigned __int128)1;
        uint64_t  hMu = (uint64_t)((one << 64) / hQ);

        cudaMemcpyToSymbol(d_Q, &hQ, sizeof(uint32_t));
        cudaMemcpyToSymbol(d_mu, &hMu, sizeof(uint64_t));
    }

    // 1) BFU 내부 상수 업로드
    {
        uint32_t om4; build_m4(OMEGA4, om4);
        cudaMemcpyToSymbol(d_m4_omega, &om4, sizeof(uint32_t));

        uint32_t m8[8]; build_m8(OMEGA8, Q, m8);
        cudaMemcpyToSymbol(d_m8, m8, sizeof(m8));

        uint32_t m16[13]; build_m16(OMEGA16, Q, m16);
        cudaMemcpyToSymbol(d_m16, m16, sizeof(m16));
    }

    // 2) 모든 플랜 생성
    auto plans = generate_all_plans(); // 순서 포함

    // 3) 디바이스 버퍼
    uint32_t* dA=nullptr;
    cudaMalloc(&dA, N*sizeof(uint32_t));

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, Q-1);

    struct Row { std::string name; float ms; };
    std::vector<Row> results;
    results.reserve(plans.size());

    // 4) 각 플랜마다 pretwiddle 사전 계산 -> runs회 커널 실행 시간 평균
    for (const auto& plan : plans){
        StagePack pack = build_stagepack_for_plan(plan);

        // 이벤트 준비
        cudaEvent_t evStart, evStop;
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);

        float total_ms = 0.0f;

        for (int r=0; r<runs; ++r){
            // 입력 생성(난수) & 업로드 (복사시간 제외)
            std::vector<uint32_t> hA(N);
            for (int i=0;i<N;++i) hA[i] = dist(rng);
            cudaMemcpy(dA, hA.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice);

            // 커널 타이밍(순수 실행만)
            cudaEventRecord(evStart);
            ntt256_kernel<<<1,128>>>(dA, pack.dstages, pack.numStages);
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            float ms=0.f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            total_ms += ms;
        }

        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);

        results.push_back({ plan_to_string(plan), total_ms / runs });

        free_stagepack(pack);
    }

    cudaFree(dA);

    // 5) 오름차순 정렬 및 출력(단위 상세)
    std::sort(results.begin(), results.end(), [](const Row& a, const Row& b){
        if (a.ms != b.ms) return a.ms < b.ms;
        return a.name < b.name;
    });

    printf("=== NTT N=256, all radix plans (factors in {2,4,8,16}, product=256) ===\n");
    printf("runs=%d, seed=0x%X (kernel time only; pretwiddle build excluded)\n\n", runs, seed);

    printf("%-20s  %12s  %12s  %12s  %14s  %12s\n",
           "plan", "avg [ms]", "avg [µs]", "avg [ns]", "ns/coeff (N=256)", "Gcoeff/s");
    for (auto& r : results){
        double ms = r.ms;
        double us = ms * 1e3;
        double ns = ms * 1e6;
        double ns_per_coeff = ns / double(N);
        double gcoeff_per_s = (double)N / (ms * 1e-3) / 1e9; // (coeff / s) -> Gcoeff/s

        printf("%-20s  %12.6f  %12.3f  %12.0f  %14.2f  %12.3f\n",
               r.name.c_str(),
               ms, us, ns, ns_per_coeff, gcoeff_per_s);
    }
    return 0;
}
