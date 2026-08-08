import time

import numpy as np

from christorch.backend.acpp import matmul


M = 1024
K = 1024
N = 1024

WARMUP = 5
RUNS = 20


rng = np.random.default_rng(42)

A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)


for _ in range(WARMUP):
    matmul(A, B)


times = []

for _ in range(RUNS):
    start = time.perf_counter()

    C = matmul(A, B)

    end = time.perf_counter()

    times.append(end - start)


average = sum(times) / len(times)

operations = 2 * M * K * N

gflops = operations / average / 1e9


print(f"AdaptiveCpp:")
print(f"Average: {average * 1000:.3f} ms")
print(f"Performance: {gflops:.3f} GFLOPS")


for _ in range(WARMUP):
    A @ B


times = []

for _ in range(RUNS):
    start = time.perf_counter()

    C_numpy = A @ B

    end = time.perf_counter()

    times.append(end - start)


average = sum(times) / len(times)

gflops = operations / average / 1e9


print()
print("NumPy:")
print(f"Average: {average * 1000:.3f} ms")
print(f"Performance: {gflops:.3f} GFLOPS")