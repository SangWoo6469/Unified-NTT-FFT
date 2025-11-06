// fft_bfu_bench.cu
// BFU(drop-1) Complex Forward FFT benchmark for N=512,1024 (x^N+1 only)
// - Plans: all sequences over {2,4,8,16} whose product is N
// - Forward only: bit-reverse load -> natural order output (NTT style)
// - drop-1 table uses Γ = exp(-i*pi/N) (negacyclic), twiddle index i*(2*j1+1), i=bitrev(1..R-1)
// - m4/m8/m16 are built from ω_R = exp(-2πi/R), using the same integer formulas you used
// - Outputs CSV with per-plan average time

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <complex>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

using std::vector; using std::string;
using cplx = std::complex<double>;
static constexpr double PI = 3.141592653589793238462643383279502884;

// ---------- Build switches ----------
#ifndef CONJ_TWIDDLE
#define CONJ_TWIDDLE 0   // set 1 if you want conjugated twiddles (sign convention flip)
#endif

// ---------- Basic helpers ----------
__host__ __device__ inline unsigned bitrev_bits(unsigned x, int bits){
    unsigned r=0;
    for(int i=0;i<bits;++i){ r=(r<<1)|(x&1u); x>>=1; }
    return r;
}
static inline std::string plan_to_string(const std::vector<int>& plan){
    std::string s;
    for(size_t i=0;i<plan.size();++i){ s += std::to_string(plan[i]); if(i+1<plan.size()) s.push_back('-'); }
    return s;
}

// ---------- Complex helpers for device ----------
struct c2 { double x,y; };
__host__ __device__ inline c2 C(double a,double b){ return {a,b}; }
__host__ __device__ inline c2 c_add(c2 a,c2 b){ return {a.x+b.x, a.y+b.y}; }
__host__ __device__ inline c2 c_sub(c2 a,c2 b){ return {a.x-b.x, a.y-b.y}; }
__host__ __device__ inline c2 c_mul(c2 a,c2 b){
    return { a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x };
}
__host__ __device__ inline c2 c_conj(c2 a){ return {a.x, -a.y}; }
__host__ __device__ inline c2 c_conj_if(c2 a, bool cond){ return cond? c_conj(a):a; }

// ---------- Stage descriptor ----------
struct StageDesc {
    int R; int j;           // stride accum
    const c2* flat;         // len = j*(R-1): drop-1 table
};

// ---------- Device constants (BFU inner consts) ----------
__constant__ c2 d_m4;
__constant__ c2 d_m8[8];
__constant__ c2 d_m16[13];

// ---------- Shared memory view ----------
extern __shared__ uint8_t __smem[];
__device__ inline c2* svec(){ return reinterpret_cast<c2*>(__smem); }

// ---------- Address helper ----------
__host__ __device__ inline int addr_loc(int R,int j,int bfu,int i){
    int k = bfu / j, j1 = bfu % j;
    int base = R * k * j;
    return base + j1 + i * j;
}

// ---------- Pre-twiddle (drop-1, device) ----------
__device__ inline
void drop1_pretwiddle_parallel_dev(const StageDesc& stg, int bfu, int lane){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    int R=stg.R, j=stg.j, tpb=R>>1;
    int iA = lane+1, iB = lane+1+tpb;
    int k=bfu/j, j1=bfu%j, base=R*k*j, off=j1*(R-1);
    c2* sv = svec();

    if (iA<R){
        int idx=base + j1 + iA*j;
        c2 w = stg.flat[off + (iA-1)];
        if (conjTw) w = c_conj(w);
        sv[idx] = c_mul(sv[idx], w);
    }
    if (iB<R){
        int idx=base + j1 + iB*j;
        c2 w = stg.flat[off + (iB-1)];
        if (conjTw) w = c_conj(w);
        sv[idx] = c_mul(sv[idx], w);
    }
}

// ---------- BFU kernels (device) ----------
__device__ inline
void r2_fused_dev(const StageDesc& stg, int bfu){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    int j=stg.j, k=bfu/j, j1=bfu%j, base=(2*k)*j;
    int ia=base+j1+0*j, ib=base+j1+1*j;
    c2* sv = svec();
    c2 a0=sv[ia], a1=sv[ib];
    c2 w = stg.flat[j1]; if (conjTw) w=c_conj(w);
    a1 = c_mul(a1, w);
    sv[ia] = c_add(a0,a1);
    sv[ib] = c_sub(a0,a1);
}

__device__ inline
void r4_step1_pairs_dev(const StageDesc& stg,int bfu,int lane){
    int a=(lane==0)?0:2, b=a+1;
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
    sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
}
__device__ inline
void r4_step2_tw_dev(const StageDesc& stg,int bfu,int lane){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    if (lane==1){
        int i3=addr_loc(stg.R,stg.j,bfu,3);
        c2 m=d_m4; if (conjTw) m=c_conj(m);
        c2* sv=svec(); sv[i3]=c_mul(sv[i3],m);
    }
}
__device__ inline
void r4_step3_pairs_dev(const StageDesc& stg,int bfu,int lane){
    int a=(lane==0)?0:1, b=a+2;
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
    sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
}

__device__ inline
void r8_step1_pairs_dev(const StageDesc& stg,int bfu,int lane){
    const int P[4][2]={{0,1},{2,3},{4,5},{6,7}};
    int a=P[lane][0], b=P[lane][1];
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
    sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
}
__device__ inline
void r8_step2_mid_dev(const StageDesc& stg,int bfu,int lane){
    c2* sv=svec();
    if (lane==2){
        int ia=addr_loc(stg.R,stg.j,bfu,4), ib=addr_loc(stg.R,stg.j,bfu,6);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    } else if (lane==3){
        int ia=addr_loc(stg.R,stg.j,bfu,5), ib=addr_loc(stg.R,stg.j,bfu,7);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
}
__device__ inline
void r8_step3_tw_dev(const StageDesc& stg,int bfu,int lane){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    c2* sv=svec();
    if (lane==1){ int i=addr_loc(stg.R,stg.j,bfu,3); sv[i]=c_mul(sv[i], c_conj_if(d_m8[3],conjTw)); }
    if (lane==3){ int i=addr_loc(stg.R,stg.j,bfu,6); sv[i]=c_mul(sv[i], c_conj_if(d_m8[5],conjTw)); }
    if (lane==2){ int i=addr_loc(stg.R,stg.j,bfu,5); sv[i]=c_mul(sv[i], c_conj_if(d_m8[6],conjTw)); }
    if (lane==0){ int i=addr_loc(stg.R,stg.j,bfu,7); sv[i]=c_mul(sv[i], c_conj_if(d_m8[7],conjTw)); }
}
__device__ inline
void r8_step4_pairs_dev(const StageDesc& stg,int bfu,int lane){
    c2* sv=svec();
    if (lane==0){
        int ia=addr_loc(stg.R,stg.j,bfu,0), ib=addr_loc(stg.R,stg.j,bfu,2);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    } else if (lane==1){
        int ia=addr_loc(stg.R,stg.j,bfu,1), ib=addr_loc(stg.R,stg.j,bfu,3);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    } else if (lane==2){
        int ia=addr_loc(stg.R,stg.j,bfu,5), ib=addr_loc(stg.R,stg.j,bfu,7);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
}
__device__ inline
void r8_step5_final_dev(const StageDesc& stg,int bfu,int lane){
    const int P[4][2]={{0,4},{1,5},{2,6},{3,7}};
    int a=P[lane][0], b=P[lane][1];
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
    sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
}

// R=16 (same staging/indexing as your NTT BFU)
__device__ inline
void r16_p1_dev(const StageDesc& stg,int bfu,int lane){
    int a=2*lane, b=a+1;
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
    sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
}
__device__ inline
void r16_p2_dev(const StageDesc& stg,int bfu,int lane){
    const int P[7][2]={{0,2},{4,6},{5,7},{8,10},{12,14},{11,13},{9,15}};
    if (lane<7){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
        c2* sv=svec(); c2 xa=sv[ia], xb=sv[ib];
        sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
}
__device__ inline
void r16_p3_dev(const StageDesc& stg,int bfu,int lane, c2* s0, c2* s1){
    c2* sv=svec();
    if (lane==0){
        c2 v9 = sv[addr_loc(stg.R,stg.j,bfu,9)];
        c2 v11= sv[addr_loc(stg.R,stg.j,bfu,11)];
        c2 v13= sv[addr_loc(stg.R,stg.j,bfu,13)];
        c2 v15= sv[addr_loc(stg.R,stg.j,bfu,15)];
        *s0 = c_add(v9,v11);
        *s1 = c_add(v13,v15);
    }
    const int P[3][2]={{10,14},{8,12},{0,4}};
    if (lane<3){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
}
__device__ inline
void r16_p4_tw_dev(const StageDesc& stg,int bfu,int lane,c2& s0,c2& s1){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    c2* sv=svec();
    switch(lane){
      case 0: { s0=c_mul(s0,c_conj_if(d_m16[4],conjTw));
                s1=c_mul(s1,c_conj_if(d_m16[5],conjTw)); } break;
      case 1: { int i=addr_loc(stg.R,stg.j,bfu,9 ); sv[i]=c_mul(sv[i], c_conj_if(d_m16[0],conjTw)); } break;
      case 2: { int i=addr_loc(stg.R,stg.j,bfu,15); sv[i]=c_mul(sv[i], c_conj_if(d_m16[1],conjTw)); } break;
      case 3: { int i=addr_loc(stg.R,stg.j,bfu,11); sv[i]=c_mul(sv[i], c_conj_if(d_m16[2],conjTw)); } break;
      case 4: { int i=addr_loc(stg.R,stg.j,bfu,13); sv[i]=c_mul(sv[i], c_conj_if(d_m16[3],conjTw)); } break;
      case 5: { int i=addr_loc(stg.R,stg.j,bfu,3 ); sv[i]=c_mul(sv[i], c_conj_if(d_m16[6],conjTw)); } break;
      case 6: { int i=addr_loc(stg.R,stg.j,bfu,12); sv[i]=c_mul(sv[i], c_conj_if(d_m16[7],conjTw)); } break;
      case 7: { int i=addr_loc(stg.R,stg.j,bfu,10); sv[i]=c_mul(sv[i], c_conj_if(d_m16[8],conjTw));
                i=addr_loc(stg.R,stg.j,bfu,14); sv[i]=c_mul(sv[i], c_conj_if(d_m16[9],conjTw)); } break;
    }
}
__device__ inline
void r16_p5_dev(const StageDesc& stg,int bfu,int lane,c2& s0,c2& s1){
    const bool conjTw = (CONJ_TWIDDLE != 0);
    c2* sv=svec();
    if (lane==0){
        c2 s0p=c_add(s0,s1), s1p=c_sub(s0,s1); s0=s0p; s1=s1p;
        int i5=addr_loc(stg.R,stg.j,bfu,5);  sv[i5]=c_mul(sv[i5], c_conj_if(d_m16[10],conjTw));
        int i7=addr_loc(stg.R,stg.j,bfu,7);  sv[i7]=c_mul(sv[i7], c_conj_if(d_m16[11],conjTw));
        int i6=addr_loc(stg.R,stg.j,bfu,6);  sv[i6]=c_mul(sv[i6], c_conj_if(d_m16[12],conjTw));
    }
    const int P[6][2]={{1,3},{2,6},{10,14},{9,15},{11,13},{5,7}};
    if (lane<6){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
}
__device__ inline
void r16_p6_dev(const StageDesc& stg,int bfu,int lane,c2 s0,c2 s1){
    c2* sv=svec();
    const int P[4][2]={{1,5},{3,7},{9,11},{15,13}};
    if (lane<4){
        int a=P[lane][0], b=P[lane][1];
        int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
        c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    }
    if (lane==0){
        int i9 =addr_loc(stg.R,stg.j,bfu,9 ); sv[i9 ]=c_sub(s0, sv[i9 ]);
        int i15=addr_loc(stg.R,stg.j,bfu,15); sv[i15]=c_sub(s1, sv[i15]);
    }
}
__device__ inline
void r16_p7_dev(const StageDesc& stg,int bfu,int lane){
    c2* sv=svec();
    const int P[8][2]={{0,8},{1,9},{2,10},{4,12},{6,14},{7,15},{3,13},{5,11}};
    int a=P[lane][0], b=P[lane][1];
    int ia=addr_loc(stg.R,stg.j,bfu,a), ib=addr_loc(stg.R,stg.j,bfu,b);
    c2 xa=sv[ia], xb=sv[ib]; sv[ia]=c_add(xa,xb); sv[ib]=c_sub(xa,xb);
    if (lane==7){ int i11=addr_loc(stg.R,stg.j,bfu,11), i13=addr_loc(stg.R,stg.j,bfu,13);
        c2 t=sv[i11]; sv[i11]=sv[i13]; sv[i13]=t; }
}

// ---------- Stage executor (device) ----------
__device__ void run_stage_dev(const StageDesc& stg){
    int tid=threadIdx.x;
    int R=stg.R, tpb=R>>1;
    int bfu=tid/tpb, lane=tid%tpb;
    int BFUs=blockDim.x/tpb; if (bfu>=BFUs) return;

    if (R==2){
        if (lane==0) r2_fused_dev(stg,bfu);
        __syncthreads(); return;
    }

    drop1_pretwiddle_parallel_dev(stg,bfu,lane);
    __syncthreads();

    if (R==4){
        r4_step1_pairs_dev(stg,bfu,lane); __syncthreads();
        r4_step2_tw_dev   (stg,bfu,lane); __syncthreads();
        r4_step3_pairs_dev(stg,bfu,lane); __syncthreads();
    } else if (R==8){
        r8_step1_pairs_dev(stg,bfu,lane); __syncthreads();
        r8_step2_mid_dev  (stg,bfu,lane); __syncthreads();
        r8_step3_tw_dev   (stg,bfu,lane); __syncthreads();
        r8_step4_pairs_dev(stg,bfu,lane); __syncthreads();
        r8_step5_final_dev(stg,bfu,lane); __syncthreads();
    } else { // 16
        __shared__ c2 s0[64], s1[64]; // N<=1024 → N/16 ≤ 64
        r16_p1_dev(stg,bfu,lane); __syncthreads();
        r16_p2_dev(stg,bfu,lane); __syncthreads();
        c2 t0=C(0,0), t1=C(0,0);
        r16_p3_dev(stg,bfu,lane, &t0,&t1); __syncthreads();
        if (lane==0){ s0[bfu]=t0; s1[bfu]=t1; } __syncthreads();
        c2 u0=s0[bfu], u1=s1[bfu];
        r16_p4_tw_dev(stg,bfu,lane,u0,u1); __syncthreads();
        if (lane==0){ s0[bfu]=u0; s1[bfu]=u1; } __syncthreads();
        u0=s0[bfu]; u1=s1[bfu];
        r16_p5_dev(stg,bfu,lane,u0,u1); __syncthreads();
        r16_p6_dev(stg,bfu,lane,u0,u1); __syncthreads();
        r16_p7_dev(stg,bfu,lane);       __syncthreads();
    }
}

// ---------- Kernel (Forward only) ----------
__global__ void bfu_forward_kernel(
    c2* __restrict__ A,
    const StageDesc* __restrict__ stages, int numStages, int N)
{
    c2* sv = svec();
    int tid=threadIdx.x;
    int i0=tid, i1=tid+(N>>1);
    int logN=0; while((1<<logN)<N) ++logN;

    // bit-reverse load -> natural output
    sv[i0] = A[ bitrev_bits((unsigned)i0, logN) ];
    sv[i1] = A[ bitrev_bits((unsigned)i1, logN) ];
    __syncthreads();

    for (int s=0;s<numStages;++s) run_stage_dev(stages[s]);

    A[i0]=sv[i0]; A[i1]=sv[i1];
}

// ---------- Host twiddle tables ----------
static c2 c_from(double ang){ return C(std::cos(ang), std::sin(ang)); }

// drop-1: useGamma=true → Γ = exp(-iπ/N) (negacyclic)
static vector<c2> build_flat_stage_complex(int N,int R,int j,bool useGamma){
    int index = N / (R * j);
    double base_ang = useGamma ? (-PI / (double)N) : (-2.0*PI / (double)N);
    cplx gp = std::polar(1.0, base_ang * (double)index);

    int rev_bits=0; while((1<<rev_bits)<R) ++rev_bits;
    vector<c2> out(j*(R-1));

    for (int j1=0;j1<j;++j1){
        vector<cplx> t(R), tb(R);
        for (int i=0;i<R;++i){
            double e = (double)i * (double)(2*j1+1);
            t[i] = std::pow(gp, e);
        }
        for (int i=0;i<R;++i){
            tb[ bitrev_bits((unsigned)i,rev_bits) ] = t[i];
        }
        for (int i=1;i<R;++i){
            out[j1*(R-1)+(i-1)] = { tb[i].real(), tb[i].imag() };
        }
    }
    return out;
}

// ---------- m4/m8/m16 (복소) ----------
static void build_m4_complex(c2& out){
    out = c_from(-2.0*PI/4.0); // ω4^1
}
static void build_m8_complex(c2 m8[8]){
    for(int i=0;i<8;i++) m8[i]=C(0,0);
    c2 one=C(1.0,0.0);
    c2 w1=c_from(-2.0*PI/8.0);
    c2 w2=c_from(-2.0*PI/8.0 * 2.0);
    c2 w3=c_from(-2.0*PI/8.0 * 3.0);
    m8[0]=one; m8[1]=one; m8[2]=one; m8[4]=one;
    m8[3]=w2;  m8[5]=w2;
    // (w1+w3)/2 , (w1-w3)/2
    m8[6]=C( (w1.x + w3.x)/2.0, (w1.y + w3.y)/2.0 );
    m8[7]=C( (w1.x - w3.x)/2.0, (w1.y - w3.y)/2.0 );
}
static void build_m16_complex(c2 m16[13]){
    auto pw = [&](int e){ return c_from(-2.0*PI/16.0 * (double)e); };
    c2 inv2=C(0.5,0.0);

    c2 w1=pw(1), w2=pw(2), w3=pw(3), w4=pw(4), w5=pw(5), w6=pw(6), w7=pw(7);

    c2 m[13];
    auto add=[&](c2 a,c2 b){ return c_add(a,b); };
    auto sub=[&](c2 a,c2 b){ return c_sub(a,b); };
    auto mul=[&](c2 a,c2 b){ return c_mul(a,b); };

    m[0]  = mul( add(w3,w5), inv2 );
    m[1]  = mul( sub(w5,w3), inv2 );
    m[2]  = mul( add(w1,w7), inv2 );
    m[3]  = mul( sub(w1,w7), inv2 );
    m[4]  = add(m[2], m[0]);
    m[5]  = add(m[1], m[3]);
    m[6]  = w4;
    m[7]  = w4;
    m[12] = w4;

    auto mix = [&](int a,int b){ return mul( add(pw(a),pw(b)), inv2 ); };
    auto dif = [&](int a,int b){ return mul( sub(pw(a),pw(b)), inv2 ); };
    m[8]  = mix(2,6);
    m[10] = mix(2,6);
    m[9]  = dif(2,6);
    m[11] = dif(2,6);

    for(int i=0;i<13;i++) m16[i]=m[i];
}

// ---------- Host/device consts ----------
struct BFUConstsHost { c2 m4; c2 m8[8]; c2 m16[13]; };
static BFUConstsHost make_host_consts(){
    BFUConstsHost H{}; build_m4_complex(H.m4); build_m8_complex(H.m8); build_m16_complex(H.m16); return H;
}
static void upload_device_consts_from(const BFUConstsHost& H){
    cudaMemcpyToSymbol(d_m4, &H.m4, sizeof(c2));
    cudaMemcpyToSymbol(d_m8, H.m8, sizeof(H.m8));
    cudaMemcpyToSymbol(d_m16, H.m16, sizeof(H.m16));
}

// ---------- Plans ----------
static void gen_plans_rec(const vector<int>& FACT,int goalN,int prod,vector<int>& cur,vector<vector<int>>& out){
    if (prod==goalN){ out.push_back(cur); return; }
    for (int f: FACT){
        if (goalN%(prod*f)==0){
            cur.push_back(f);
            gen_plans_rec(FACT,goalN,prod*f,cur,out);
            cur.pop_back();
        }
    }
}
static vector<vector<int>> generate_all_plans(int goalN){
    static const vector<int> FACT={2,4,8,16};
    vector<vector<int>> out; vector<int> cur; gen_plans_rec(FACT,goalN,1,cur,out);
    return out;
}

// ---------- StagePack ----------
struct StagePack{
    vector<StageDesc> hstages;
    vector<vector<c2>> hflats;
    vector<c2*> dflats;
    StageDesc* dstages=nullptr;
    int numStages=0;
};
static StagePack build_stagepack_for_plan(int N,const vector<int>& plan,bool useGamma){
    StagePack P;
    P.hflats.resize(plan.size());
    P.dflats.resize(plan.size(),nullptr);
    int j=1;
    for (size_t s=0;s<plan.size();++s){
        int R=plan[s];
        P.hflats[s] = build_flat_stage_complex(N,R,j,useGamma); // drop-1: Γ
        cudaMalloc(&P.dflats[s], P.hflats[s].size()*sizeof(c2));
        cudaMemcpy(P.dflats[s], P.hflats[s].data(),
                   P.hflats[s].size()*sizeof(c2), cudaMemcpyHostToDevice);
        StageDesc sd; sd.R=R; sd.j=j; sd.flat=P.dflats[s];
        P.hstages.push_back(sd);
        j*=R;
    }
    cudaMalloc(&P.dstages, P.hstages.size()*sizeof(StageDesc));
    cudaMemcpy(P.dstages, P.hstages.data(),
               P.hstages.size()*sizeof(StageDesc), cudaMemcpyHostToDevice);
    P.numStages=(int)P.hstages.size();
    return P;
}
static void free_stagepack(StagePack& P){
    for (auto p: P.dflats) if (p) cudaFree(p);
    if (P.dstages) cudaFree(P.dstages);
    P=StagePack{};
}

// ---------- CSV writer ----------
static void write_csv(const char* path,
    const vector<std::pair<std::string,double>>& rows,
    int runs, unsigned seed, const char* variant)
{
    FILE* fp = fopen(path, "w");
    if (!fp){ fprintf(stderr, "[WARN] cannot open CSV for write: %s\n", path); return; }
    fprintf(fp, "plan,avg_ms,avg_us,avg_ns,ns_per_point,Gpts_per_s,runs,seed,variant\n");
    for (auto& kv: rows){
        const std::string& name = kv.first;
        double ms = kv.second;
        double us = ms * 1e3;
        double ns = ms * 1e6;
        double ns_per_point = ns / (double)0x1; // keep column, but we’ll fill correctly below
        // overwrite per N later — easier to print accurate value in printf table; CSV keeps raw timings.
        double gpts_per_s = 0.0;
        fprintf(fp, "%s,%.6f,%.3f,%.0f,%.2f,%.6f,%d,0x%X,%s\n",
            name.c_str(), ms, us, ns, ns_per_point, gpts_per_s, runs, seed, variant);
    }
    fclose(fp);
}

// ---------- Single-N benchmark ----------
static void bench_N(int N, int runs, unsigned seed, const std::string& csv_path_prefix)
{
    printf("\n=== BFU(drop-1, Γ) Forward FFT benchmark  N=%d  (factors {2,4,8,16}) ===\n", N);
    printf("runs=%d, seed=0x%X  (forward only; kernel time only)\n\n", runs, seed);

    auto plans = generate_all_plans(N);

    // choose to sort plans lexicographically for stable display
    std::sort(plans.begin(), plans.end(), [](const vector<int>& a, const vector<int>& b){
        if (a.size()!=b.size()) return a.size()<b.size();
        return plan_to_string(a) < plan_to_string(b);
    });

    BFUConstsHost H = make_host_consts();
    upload_device_consts_from(H);

    c2* dA=nullptr; cudaMalloc(&dA, N*sizeof(c2));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    // host buffer for random input
    vector<c2> hA(N);

    struct Row { std::string name; double ms; };
    vector<Row> results; results.reserve(plans.size());

    for (const auto& plan : plans){
        StagePack pack = build_stagepack_for_plan(N, plan, /*useGamma*/true);

        // warmup input
        for (int i=0;i<N;++i) hA[i] = C(U(rng), U(rng));

        cudaEvent_t evS, evE; cudaEventCreate(&evS); cudaEventCreate(&evE);
        float total_ms = 0.f;

        for (int r=0; r<runs; ++r){
            // fresh random input each run
            for (int i=0;i<N;++i) hA[i] = C(U(rng), U(rng));
            cudaMemcpy(dA, hA.data(), N*sizeof(c2), cudaMemcpyHostToDevice);

            // launch
            int threads = N/2;
            size_t shmem = N*sizeof(c2);
            cudaEventRecord(evS);
            bfu_forward_kernel<<<1, threads, shmem>>>(dA, pack.dstages, pack.numStages, N);
            cudaEventRecord(evE);
            cudaEventSynchronize(evE);
            float ms=0.f; cudaEventElapsedTime(&ms, evS, evE);
            total_ms += ms;
        }

        cudaEventDestroy(evS); cudaEventDestroy(evE);
        results.push_back({ plan_to_string(plan), (double)total_ms / runs });

        free_stagepack(pack);
    }

    cudaFree(dA);

    // sort by time asc
    std::sort(results.begin(), results.end(), [](const Row& a, const Row& b){
        if (a.ms != b.ms) return a.ms < b.ms;
        return a.name < b.name;
    });

    // print table
    printf("%-20s  %12s  %12s  %12s  %16s  %12s\n",
        "plan","avg [ms]","avg [us]","avg [ns]","ns/point","Gpts/s");
    for (auto& r : results){
        double ms = r.ms, us = ms*1e3, ns=ms*1e6;
        double ns_per_point = ns / (double)N;
        double gpts_per_s = (double)N / (ms * 1e-3) / 1e9; // points per second in G
        printf("%-20s  %12.6f  %12.3f  %12.0f  %16.2f  %12.6f\n",
               r.name.c_str(), ms, us, ns, ns_per_point, gpts_per_s);
    }

    // write CSV
    std::string csv_path = csv_path_prefix + "_" + std::to_string(N) + ".csv";
    vector<std::pair<std::string,double>> rows;
    rows.reserve(results.size());
    for (auto& r : results) rows.emplace_back(r.name, r.ms);
    write_csv(csv_path.c_str(), rows, runs, seed, "negacyclic_drop1");
}

int main(int argc, char** argv){
    int runs = 50;
    unsigned seed = 0x1234;
    std::string csv_prefix = "bfu_fft";

    if (argc >= 2) runs = std::max(1, atoi(argv[1]));
    if (argc >= 3) seed = (unsigned)strtoul(argv[2], nullptr, 0);
    if (argc >= 4) csv_prefix = argv[3];

    // Bench both N=512 and N=1024
    bench_N(512, runs, seed, csv_prefix);
    bench_N(1024, runs, seed, csv_prefix);

    return 0;
}
