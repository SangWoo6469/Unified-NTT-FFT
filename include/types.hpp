#pragma once
#include <cstdint>
#include <cmath>
#include "config.hpp"

#if !defined(__CUDACC__) && !defined(__CUDA_RUNTIME_H__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __constant__
#define __constant__
#endif
// Minimal double2 for host-only compilation
struct double2 { double x; double y; };
inline double2 make_double2(double x, double y){ return double2{x,y}; }
#endif

#if USE_NTT
using elem_t = uint32_t;
#else
struct cplx { double x, y; };
using elem_t = cplx;
__host__ __device__ inline cplx cadd(cplx a, cplx b){ return {a.x+b.x, a.y+b.y}; }
__host__ __device__ inline cplx csub(cplx a, cplx b){ return {a.x-b.x, a.y-b.y}; }
__host__ __device__ inline cplx cmul(cplx a, cplx b){ return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x}; }
#endif

// Accurate Barrett reduction (k=64)
__host__ __device__ static inline uint32_t barrett_reduce(uint64_t a){
    constexpr uint64_t MU = (~(uint64_t)0) / (uint64_t)Q; // floor(2^64 / Q)
#if defined(__CUDA_ARCH__)
    uint64_t qhat = __umul64hi(a, MU);
#else
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)MU;
    uint64_t qhat = (uint64_t)(prod >> 64);
#endif
    uint64_t r = a - qhat * (uint64_t)Q;
    if(r >= Q) r -= Q;
    if(r >= Q) r -= Q;
    return (uint32_t)r;
}

// NTT modular ops (no reductions/comparisons for add/sub)
__host__ __device__ inline uint32_t add_mod(uint32_t a, uint32_t b){
    return (uint32_t)(a + b);
}
__host__ __device__ inline uint32_t sub_mod(uint32_t a, uint32_t b){
    return (uint32_t)(a - b);
}
__host__ __device__ inline uint32_t mul_mod(uint32_t a, uint32_t b){
    uint64_t z = (uint64_t)a * (uint64_t)b;
    return barrett_reduce(z);
}
