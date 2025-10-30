# UNIFIED NTT/FFT

본 저장소는 NTT/FFT 통합 프레임워크의 스켈레톤입니다.

- `src/main_fft.cpp`: Falcon 검증/데모 엔트리
- `src/main_ntt.cpp`: Dilithium 검증/데모 엔트리
- `include/`: 공개 헤더
- `third_party/`: (선택) 외부 테이블/헤더

빌드 예시:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
```
