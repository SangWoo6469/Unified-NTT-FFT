cmake_minimum_required(VERSION 3.21)

# -----------------------
# Project & Languages
# -----------------------
project(cuda_bfu_ntt_fft LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------
# Options
# -----------------------
# N: 512 or 1024
set(N "512" CACHE STRING "Transform size N (512 or 1024)")
set_property(CACHE N PROPERTY STRINGS 512 1024)

# BATCH: number of polynomials per kernel launch (FFT path에서만 주로 사용)
set(BATCH "256" CACHE STRING "Batch size for GPU execution")

# USE_NTT: ON(1)=NTT/Dilithium path, OFF(0)=FFT/Falcon path
option(USE_NTT "Build NTT path (otherwise FFT path)" ON)

# USE_DILITHIUM_ZETAS: Dilithium 공식 zetas 테이블 사용 여부(선택)
option(USE_DILITHIUM_ZETAS "Use Dilithium official zetas for NTT" OFF)

# FAST_MATH: (선택) FP 빠른 연산 최적화
option(USE_FAST_MATH "Enable CUDA fast-math for FFT path" OFF)

# Compute capability (필요한 것만 남기거나 추가)
set(CMAKE_CUDA_ARCHITECTURES 86 89 90 CACHE STRING "CUDA arch list")

# -----------------------
# Build Type
# -----------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------
# Compile definitions (global)
# -----------------------
add_compile_definitions(N=${N})
add_compile_definitions(BATCH=${BATCH})
if (USE_NTT)
  add_compile_definitions(USE_NTT=1)
else()
  add_compile_definitions(USE_NTT=0)
endif()
if (USE_DILITHIUM_ZETAS)
  add_compile_definitions(USE_DILITHIUM_ZETAS=1)
else()
  add_compile_definitions(USE_DILITHIUM_ZETAS=0)
endif()

# -----------------------
# Include paths
# -----------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# -----------------------
# Warnings / Flags
# -----------------------
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

if (USE_FAST_MATH)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# -----------------------
# Sources (shared objects)
# -----------------------
add_library(core_objs OBJECT
  src/kernels.cu
  src/host_precompute.cpp
  src/reference_cpu.cpp
)
set_target_properties(core_objs PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------
# Executables
# -----------------------
# FFT 테스트 드라이버 (Falcon, USE_NTT=0에서 실행)
add_executable(run_fft
  src/main_fft.cpp
  $<TARGET_OBJECTS:core_objs>
)
set_target_properties(run_fft PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# NTT 테스트 드라이버 (Dilithium, USE_NTT=1에서 실행)
add_executable(run_ntt
  src/main_ntt.cpp
  $<TARGET_OBJECTS:core_objs>
)
set_target_properties(run_ntt PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------
# Linking (cudart 등)
# -----------------------
find_package(CUDAToolkit REQUIRED)
# core_objs는 OBJECT라 링크 불필요, 실행파일만 cudart 링크
target_link_libraries(run_fft PRIVATE CUDA::cudart)
target_link_libraries(run_ntt PRIVATE CUDA::cudart)


