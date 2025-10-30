#pragma once
#include <cstdint>

#ifndef N
#define N 256
#endif

#ifndef BATCH
#define BATCH 1
#endif

#ifndef USE_NTT
#define USE_NTT 1
#endif

#ifndef USE_DILITHIUM_ZETAS
#define USE_DILITHIUM_ZETAS 0
#endif

#ifndef Q
#define Q 8380417u
#endif

#if N==256
#define MAX_STAGES 3
static constexpr int PLAN[MAX_STAGES] = {16,4,4};
#elif N==512
#define MAX_STAGES 3
static constexpr int PLAN[MAX_STAGES] = {16,8,4};
#elif N==1024
#define MAX_STAGES 3
static constexpr int PLAN[MAX_STAGES] = {16,8,8};
#else
#error "Unsupported N"
#endif
