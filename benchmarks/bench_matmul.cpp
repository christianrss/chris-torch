#include "matmul.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>

int main()
{
    constexpr std::size_t M = 1024;
    constexpr std::size_t K = 1024;
    constexpr std::size_t N = 1024;

    constexpr int warmup_runs = 5;
    constexpr int benchmark_runs = 20;

    sycl::queue q{
        sycl::property::queue::in_order{}
    };

    std::cout
        << "Device: "
        << q.get_device()
                .get_info<sycl::info::device::name>()
        << '\n';

    float* A = sycl::malloc_shared<float>(M * K, q);
    float* B = sycl::malloc_shared<float>(K * N, q);
    float* C = sycl::malloc_shared<float>(M * N, q);

    if(!A || !B || !C)
    {
        std::cerr << "USM allocation failed\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(std::size_t i = 0; i < M * K; ++i)
        A[i] = dist(rng);

    for(std::size_t i = 0; i < K * N; ++i)
        B[i] = dist(rng);

    // Warm-up
    for(int i = 0; i < warmup_runs; ++i)
    {
        matmul(q, A, B, C, M, K, N);
    }

    q.wait_and_throw();

    double total_ms = 0.0;

    for(int i = 0; i < benchmark_runs; ++i)
    {
        auto start = std::chrono::steady_clock::now();

        matmul(q, A, B, C, M, K, N);

        q.wait_and_throw();

        auto end = std::chrono::steady_clock::now();

        const double ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        total_ms += ms;

        std::cout
            << "Run "
            << i + 1
            << ": "
            << ms
            << " ms\n";
    }

    const double avg_ms =
        total_ms / benchmark_runs;

    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(K) *
        static_cast<double>(N);

    const double gflops =
        operations /
        (avg_ms / 1000.0) /
        1e9;

    std::cout << "\n";
    std::cout << "Matrix size: "
        << M << "x" << K
        << " * "
        << K << "x" << N
        << '\n';

    std::cout
        << "Average: "
        << avg_ms
        << " ms\n";

    std::cout
        << "Performance: "
        << gflops
        << " GFLOPS\n";

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return 0;
}