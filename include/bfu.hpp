#pragma once
#include "types.hpp"

// === 상수 메모리의 BFU m[] ===
#if USE_NTT
extern __constant__ uint32_t d_bfu_m4[1];
extern __constant__ uint32_t d_bfu_m8[8];
extern __constant__ uint32_t d_bfu_m16[18];
#else
extern __constant__ double2  d_bfu_m4_c[1];
extern __constant__ double2  d_bfu_m8_c[8];
extern __constant__ double2  d_bfu_m16_c[18];
#endif

// In-place Hadamard with single temporary
__device__ inline void had2(uint32_t& u, uint32_t& v){
    uint32_t t = u;
    u = add_mod(u, v);
    v = sub_mod(t, v);
}
#if !USE_NTT
__device__ inline void had2(cplx& u, cplx& v){
    cplx t = u;
    u = cadd(u, v);
    v = csub(t, v);
}
#endif

// --- Radix-2 ---
__device__ inline void BFU_radix2(elem_t* v){
    had2(v[0], v[1]);
}

// --- Radix-4 (in-place hadamard schedule) ---
__device__ inline void BFU_radix4(elem_t* a){
#if USE_NTT
    // stage 1
    had2(a[0], a[1]);
    had2(a[2], a[3]);
    // apply twiddle on lane-3 as per wiring
    a[3] = mul_mod(a[3], d_bfu_m4[0]);
    // stage 2
    had2(a[0], a[2]);
    had2(a[1], a[3]);
#else
    // stage 1
    had2(a[0], a[1]);
    had2(a[2], a[3]);
    // stage 2
    had2(a[0], a[2]);
    had2(a[1], a[3]);
#endif
}

// --- Radix-8 (in-place schedule matching Python wiring) ---
__device__ inline void BFU_radix8(elem_t* a){
#if USE_NTT
    // Step 1
    had2(a[0], a[1]); had2(a[2], a[3]); had2(a[4], a[5]); had2(a[6], a[7]);
    // Step 2
    had2(a[4], a[6]); had2(a[5], a[7]);
    // Step 3: twiddle multiplies
    a[3] = mul_mod(a[3], d_bfu_m8[3]);
    a[6] = mul_mod(a[6], d_bfu_m8[5]);
    a[5] = mul_mod(a[5], d_bfu_m8[6]);
    a[7] = mul_mod(a[7], d_bfu_m8[7]);
    // Step 4
    had2(a[0], a[2]); had2(a[1], a[3]); had2(a[5], a[7]);
    // Step 5
    had2(a[0], a[4]); had2(a[1], a[5]); had2(a[2], a[6]); had2(a[3], a[7]);
#else
    // Step 1
    had2(a[0], a[1]); had2(a[2], a[3]); had2(a[4], a[5]); had2(a[6], a[7]);
    // Step 2
    had2(a[4], a[6]); had2(a[5], a[7]);
    // Step 3
    { double2 t=d_bfu_m8_c[3]; a[3]=cmul(a[3], {t.x,t.y}); }
    { double2 t=d_bfu_m8_c[5]; a[6]=cmul(a[6], {t.x,t.y}); }
    { double2 t=d_bfu_m8_c[6]; a[5]=cmul(a[5], {t.x,t.y}); }
    { double2 t=d_bfu_m8_c[7]; a[7]=cmul(a[7], {t.x,t.y}); }
    // Step 4
    had2(a[0], a[2]); had2(a[1], a[3]); had2(a[5], a[7]);
    // Step 5
    had2(a[0], a[4]); had2(a[1], a[5]); had2(a[2], a[6]); had2(a[3], a[7]);
#endif
}

// --- Radix-16 (in-place schedule matching provided wiring) ---
__device__ inline void BFU_radix16(elem_t* a){
#if USE_NTT
    // Stage 1
    for(int i=0;i<16;i+=2) had2(a[i], a[i+1]);
    // Stage 2
    had2(a[0], a[2]); had2(a[4], a[6]); had2(a[5], a[7]); had2(a[8], a[10]); had2(a[12], a[14]); had2(a[11], a[13]); had2(a[9], a[15]);
    // Stage 3
    uint32_t s0 = add_mod(a[9],a[11]);
    uint32_t s1 = add_mod(a[13],a[15]);
    had2(a[10], a[14]); had2(a[8], a[12]); had2(a[0], a[4]);
    a[9]  = mul_mod(a[9] , d_bfu_m16[0]);
    a[15] = mul_mod(a[15], d_bfu_m16[1]);
    a[11] = mul_mod(a[11], d_bfu_m16[2]);
    a[13] = mul_mod(a[13], d_bfu_m16[3]);
    s0    = mul_mod(s0   , d_bfu_m16[4]);
    s1    = mul_mod(s1   , d_bfu_m16[5]);
    a[3]  = mul_mod(a[3] , d_bfu_m16[6]);
    a[12] = mul_mod(a[12], d_bfu_m16[7]);
    a[10] = mul_mod(a[10], d_bfu_m16[8]);
    a[14] = mul_mod(a[14], d_bfu_m16[9]);
    a[5]  = mul_mod(a[5] , d_bfu_m16[10]);
    a[7]  = mul_mod(a[7] , d_bfu_m16[11]);
    a[6]  = mul_mod(a[6] , d_bfu_m16[12]);
    had2(s0, s1);
    had2(a[1], a[3]); had2(a[2], a[6]); had2(a[10], a[14]); had2(a[9], a[15]); had2(a[11], a[13]); had2(a[5], a[7]);
    had2(a[1], a[5]); had2(a[3], a[7]); had2(a[9], a[11]); had2(a[15], a[13]);
    a[9]  = sub_mod(s0, a[9]);
    a[15] = sub_mod(s1, a[15]);
    // Output mix
    had2(a[0], a[8]); had2(a[1], a[9]); had2(a[2], a[10]); had2(a[3], a[13]); had2(a[4], a[12]); had2(a[5], a[11]); had2(a[6], a[14]); had2(a[7], a[15]);
    { uint32_t t=a[13]; a[13]=a[11]; a[11]=t; }
#else
    // Stage 1
    for(int i=0;i<16;i+=2) had2(a[i], a[i+1]);
    // Stage 2
    had2(a[0], a[2]); had2(a[4], a[6]); had2(a[5], a[7]); had2(a[8], a[10]); had2(a[12], a[14]); had2(a[11], a[13]); had2(a[9], a[15]);
    // Stage 3 and twiddles
    {
      cplx s0 = cadd(a[9],a[11]);
      cplx s1 = cadd(a[13],a[15]);
      had2(a[10], a[14]); had2(a[8], a[12]); had2(a[0], a[4]);
      auto mc=[&](cplx z, double2 t){ return cmul(z,{t.x,t.y}); };
      a[9]  = mc(a[9] , d_bfu_m16_c[0]);
      a[15] = mc(a[15], d_bfu_m16_c[1]);
      a[11] = mc(a[11], d_bfu_m16_c[2]);
      a[13] = mc(a[13], d_bfu_m16_c[3]);
      s0    = mc(s0    , d_bfu_m16_c[4]);
      s1    = mc(s1    , d_bfu_m16_c[5]);
      a[3]  = mc(a[3] ,  d_bfu_m16_c[6]);
      a[12] = mc(a[12], d_bfu_m16_c[7]);
      a[10] = mc(a[10], d_bfu_m16_c[8]);
      a[14] = mc(a[14], d_bfu_m16_c[9]);
      a[5]  = mc(a[5] ,  d_bfu_m16_c[10]);
      a[7]  = mc(a[7] ,  d_bfu_m16_c[11]);
      a[6]  = mc(a[6] ,  d_bfu_m16_c[12]);
      had2(s0, s1);
      had2(a[1], a[3]); had2(a[2], a[6]); had2(a[10], a[14]); had2(a[9], a[15]); had2(a[11], a[13]); had2(a[5], a[7]);
      had2(a[1], a[5]); had2(a[3], a[7]); had2(a[9], a[11]); had2(a[15], a[13]);
      a[9]  = csub(s0, a[9]);
      a[15] = csub(s1, a[15]);
    }
    // Output mix
    had2(a[0], a[8]); had2(a[1], a[9]); had2(a[2], a[10]); had2(a[3], a[13]); had2(a[4], a[12]); had2(a[5], a[11]); had2(a[6], a[14]); had2(a[7], a[15]);
    { cplx t=a[13]; a[13]=a[11]; a[11]=t; }
#endif
}
