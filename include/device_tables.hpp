#pragma once
#include "config.hpp"
#include "types.hpp"

#if USE_NTT
extern __constant__ uint32_t d_stage_twiddles_ntt[MAX_STAGES * N];
extern __constant__ uint32_t d_stage_twiddles_ntt_inv[MAX_STAGES * N];
#else
extern __constant__ double2  d_stage_twiddles_fft[MAX_STAGES * N];
extern __constant__ double2  d_stage_twiddles_fft_inv[MAX_STAGES * N];
#endif
