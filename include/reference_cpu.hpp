#pragma once
#include <vector>
#include <complex>
#include <cstdint>
#include "config.hpp"

void cpu_fft_forward_radix2(const std::vector<std::complex<double>>& x,
                            std::vector<std::complex<double>>& out);
void cpu_ifft_radix2(const std::vector<std::complex<double>>& X,
                     std::vector<std::complex<double>>& x);

void cpu_ntt_forward(const std::vector<uint32_t>& a,
                     std::vector<uint32_t>& A);

void cpu_schoolbook_negacyclic_fft(const std::vector<std::complex<double>>& a,
                                   const std::vector<std::complex<double>>& b,
                                   std::vector<std::complex<double>>& c);
void cpu_schoolbook_negacyclic_ntt(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b,
                                   std::vector<uint32_t>& c);
