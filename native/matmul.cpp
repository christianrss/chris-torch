#include "matmul.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstring>
#include <exception>

void matmul(
    sycl::queue& q,
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
)
{
    q.parallel_for(
        sycl::range<2>{M, N},
        [=](sycl::id<2> idx)
        {
            const std::size_t row = idx[0];
            const std::size_t col = idx[1];

            float sum = 0.0f;

            for(std::size_t k = 0; k < K; ++k)
            {
                sum += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    );
}

extern "C"
int ct_matmul_f32(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
)
{
    try
    {
        static sycl::queue q{
            sycl::property::queue::in_order{}
        };

        float* A_sycl = sycl::malloc_shared<float>(M * K, q);
        float* B_sycl = sycl::malloc_shared<float>(K * N, q);
        float* C_sycl = sycl::malloc_shared<float>(M * N, q);

        if(!A_sycl || !B_sycl || !C_sycl)
        {
            if(A_sycl) sycl::free(A_sycl, q);
            if(B_sycl) sycl::free(B_sycl, q);
            if(C_sycl) sycl::free(C_sycl, q);

            return 1;
        }

        std::memcpy(A_sycl, A, M * K * sizeof(float));
        std::memcpy(B_sycl, B, K * N * sizeof(float));

        matmul(
            q,
            A_sycl,
            B_sycl,
            C_sycl,
            M,
            K,
            N
        );

        q.wait_and_throw();

        std::memcpy(C, C_sycl, M * N * sizeof(float));

        sycl::free(A_sycl, q);
        sycl::free(B_sycl, q);
        sycl::free(C_sycl, q);

        return 0;
    }
    catch(const std::exception&)
    {
        return 2;
    }
}