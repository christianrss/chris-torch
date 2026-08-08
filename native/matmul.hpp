#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>

void matmul(
    sycl::queue& q,
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
);

extern "C" int ct_matmul_f32(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
);