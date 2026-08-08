# chris-torch

A PyTorch-like deep learning framework written from scratch for learning and experimenting with autograd, tensor operations, neural networks, and heterogeneous computing.

Chris Torch currently implements its core autograd system in Python/NumPy and is experimenting with native C++/SYCL kernels through AdaptiveCpp.

> This is an educational and experimental project. It is not intended to replace PyTorch.

## Goals

The project is built from scratch to explore the internals of modern deep learning frameworks, including:

* automatic differentiation;
* computational graphs;
* tensor operations;
* broadcasting;
* gradient propagation;
* neural network primitives;
* native numerical kernels;
* heterogeneous CPU/GPU computing;
* backend abstraction;
* kernel optimization.

The long-term goal is to progressively move performance-critical operations from NumPy to native heterogeneous kernels while keeping the high-level framework and autograd system in Python.

## Current architecture

```text
Chris Torch
    │
    ├── Python
    │   │
    │   ├── Var
    │   ├── Function
    │   ├── Autograd
    │   └── Neural network operations
    │
    ├── NumPy backend
    │   │
    │   └── Reference tensor operations
    │
    └── AdaptiveCpp backend
        │
        ├── C++17
        ├── SYCL
        ├── USM
        └── Native kernels
```

The NumPy implementation provides a simple reference backend while selected operations can be implemented as native SYCL kernels.

## Autograd

Chris Torch implements a small reverse-mode automatic differentiation engine.

The core abstraction is based on variables and functions:

```text
Var
 │
 ▼
Function
 │
 ▼
Var
 │
 ▼
Function
 │
 ▼
Var
```

During the forward pass, operations construct the computational graph.

During `backward()`, Chris Torch traverses this graph in reverse order and propagates gradients through each operation.

Currently implemented operations include concepts such as:

* addition;
* subtraction;
* multiplication;
* division;
* power;
* exponential;
* sine;
* reshape;
* transpose;
* sum;
* matrix multiplication.

Broadcasted operations also reduce gradients back to the original input shape when required.

## AdaptiveCpp / SYCL backend

Chris Torch includes an experimental native backend built with [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp).

The architecture is:

```text
Python / NumPy
      │
      ▼
Python native binding
      │
      ▼
libchristorch_acpp.so
      │
      ▼
C++ / SYCL kernels
      │
      ▼
AdaptiveCpp Runtime
      │
      └── execution backend
```

The native code is compiled into:

```text
build/libchristorch_acpp.so
```

and can be called from Python.

This allows performance-critical tensor operations to gradually move from:

```python
np.dot(x, W)
```

toward native SYCL implementations.

## Native MatMul

The first native numerical kernel is matrix multiplication.

For matrices:

```text
A[M × K]
B[K × N]
```

Chris Torch computes:

```text
C[M × N] = A × B
```

The initial implementation is intentionally simple: each SYCL work-item computes one output element.

Conceptually:

```cpp
q.parallel_for(
    sycl::range<2>{M, N},
    [=](sycl::id<2> idx)
    {
        const std::size_t row = idx[0];
        const std::size_t col = idx[1];

        float sum = 0.0f;

        for(std::size_t k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
);
```

This implementation currently serves as a correctness baseline for future optimization work.

## Native backend structure

```text
chris-torch/
├── native/
│   ├── matmul.hpp
│   └── matmul.cpp
│
├── benchmarks/
│   └── bench_matmul.cpp
│
├── build/
│   ├── libchristorch_acpp.so
│   └── benchmarks/
│       └── bench_matmul
│
└── Makefile
```

Additional native kernels can be added under `native/`.

Planned examples include:

```text
matmul
add
mul
relu
softmax
reductions
normalization
```

## Building the AdaptiveCpp backend

AdaptiveCpp must be installed and available to the build system.

The current development environment uses:

```text
/etc/acpp/bin/acpp
```

Build the native backend:

```bash
make
```

This generates:

```text
build/libchristorch_acpp.so
```

Clean generated files:

```bash
make clean
```

Rebuild:

```bash
make rebuild
```

## Benchmarking

Native kernels can be benchmarked independently from Python.

Run:

```bash
make benchmark
```

### Current MatMul baseline

Environment:

```text
Backend: AdaptiveCpp OpenMP host device
Matrix:  1024 × 1024 × 1024
Kernel:  naive SYCL MatMul
Runs:    20
```

Current baseline:

```text
Average:     ~362.7 ms
Performance: ~5.92 GFLOPS
```

These numbers represent the current naive implementation and are intended as a baseline for measuring future optimizations.

They should not be interpreted as the expected performance limit of AdaptiveCpp or SYCL.

## MatMul optimization roadmap

The current implementation intentionally starts from a simple kernel.

Future experiments can progressively introduce:

```text
Naive MatMul
      │
      ▼
nd_range
      │
      ▼
Work-group tiling
      │
      ▼
Local memory
      │
      ▼
Vectorization
      │
      ▼
Memory-access optimization
      │
      ▼
CPU/GPU-specific tuning
```

Each optimization can be compared against the original naive implementation using the benchmark suite.

## Backend evolution

The current native interface still moves data between NumPy-managed memory and SYCL-managed memory.

Conceptually:

```text
NumPy
  │
  │ copy
  ▼
SYCL USM
  │
  ▼
Kernel
  │
  │ copy
  ▼
NumPy
```

This is intentionally a first step.

A more advanced tensor backend could eventually own native memory directly:

```text
Tensor
  │
  ├── shape
  ├── dtype
  ├── device
  └── native memory
          │
          ▼
     SYCL kernels
```

This would allow tensors to remain on the execution device across multiple operations instead of copying data for every kernel invocation.

## Planned development

The project is gradually exploring:

```text
Autograd
   │
   ├── computational graphs
   ├── reverse-mode differentiation
   └── gradient checking
   │
   ▼
Tensor operations
   │
   ├── broadcasting
   ├── reshape
   ├── transpose
   ├── reductions
   └── matrix multiplication
   │
   ▼
Neural networks
   │
   ├── MLP
   ├── CNN
   └── RNN
   │
   ▼
Native backend
   │
   ├── C++
   ├── SYCL
   └── AdaptiveCpp
   │
   ▼
Kernel optimization
   │
   ├── parallel execution
   ├── memory optimization
   ├── tiling
   └── vectorization
   │
   ▼
Heterogeneous computing
       │
       ├── CPU
       └── GPU
```

## Philosophy

Chris Torch is intentionally built from fundamental components rather than wrapping an existing deep learning framework.

The project prioritizes understanding:

* how automatic differentiation works;
* how tensor operations are implemented;
* how gradients propagate through computational graphs;
* how numerical kernels map onto parallel hardware;
* how Python frameworks interact with native code;
* how heterogeneous runtimes execute kernels;
* how low-level optimization affects deep learning workloads.

## Status

Chris Torch is under active development.

Current work includes:

* Python/NumPy autograd engine;
* core differentiable operations;
* broadcasting support;
* gradient checking;
* matrix multiplication;
* experimental AdaptiveCpp/SYCL backend;
* native shared-library integration;
* native MatMul benchmarking;
* CPU execution through the AdaptiveCpp OpenMP backend.

GPU execution and more advanced kernel optimizations are future work.

## License

This project is intended for educational, research, and experimental purposes.
