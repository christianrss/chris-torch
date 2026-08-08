# Chris Torch

A PyTorch-like deep learning framework written from scratch for studying automatic differentiation, tensor operations, neural networks, numerical kernels, and heterogeneous computing.

Chris Torch currently implements its core autograd system in Python/NumPy and includes an experimental native C++/SYCL backend built with [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp).

> Experimental and educational project. Chris Torch is not intended to replace PyTorch.

## Overview

Chris Torch is an attempt to understand deep learning frameworks from the inside out rather than treating tensor libraries and automatic differentiation as black boxes.

The project explores multiple layers of the machine learning software stack:

```text
Neural Networks
       │
       ▼
Automatic Differentiation
       │
       ▼
Tensor Operations
       │
       ▼
Native Numerical Kernels
       │
       ▼
SYCL / AdaptiveCpp
       │
       ▼
CPU / GPU
```

The framework starts with simple NumPy implementations and progressively moves performance-critical operations toward native C++/SYCL kernels.

## Related Projects

Chris Torch is part of a collection of personal projects exploring machine learning systems from training frameworks down to model execution and heterogeneous computing.

### Chris Torch

PyTorch-like framework implementing automatic differentiation, tensor operations, neural network primitives, and experimental native compute backends.

This repository.

### Chris GPT

GPT-2 implementation and training experiments intended to explore transformer training using the Chris Torch ecosystem.

### [Chris Llama](https://github.com/christianrss/chris-llama)

LLM inference engine written in C.

The project explores low-level model execution, GGUF, tensor operations, KV cache, quantization, and other techniques used by modern LLM inference engines.

### tests-acpp

Experimental repository for studying AdaptiveCpp, SYCL, heterogeneous computing, numerical kernels, benchmarking, and compiler/runtime behavior.

It is also used as a testbed while learning and experimenting with the AdaptiveCpp ecosystem.

### [AdaptiveCpp Fork](https://github.com/christianrss/AdaptiveCpp)

Personal fork of AdaptiveCpp used for experiments, development branches, bug investigation, and potential upstream contributions.

Official upstream project:

[AdaptiveCpp/AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)

## Goals

Chris Torch is built from fundamental components to explore:

* reverse-mode automatic differentiation;
* computational graphs;
* gradient propagation;
* tensor operations;
* broadcasting;
* matrix operations;
* neural network primitives;
* native C++ kernels;
* SYCL programming;
* heterogeneous CPU/GPU execution;
* backend abstraction;
* kernel optimization;
* interaction between Python and native code.

The long-term direction is to progressively replace selected NumPy operations with native kernels while preserving a high-level Python interface.

## Current Architecture

```text
                    Chris Torch
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
     Python Core                  Native Backend
          │                             │
    ┌─────┴─────┐                       ▼
    │           │                  C++ / SYCL
    ▼           ▼                       │
  Var       Function                    ▼
    │           │                  AdaptiveCpp
    └─────┬─────┘                       │
          │                       ┌─────┴─────┐
          ▼                       ▼           ▼
       Autograd                  CPU         GPU
          │
          ▼
       NumPy
```

At the moment, NumPy remains the reference implementation for most operations.

Selected operations can be moved into the experimental AdaptiveCpp backend.

## Autograd Engine

The core abstraction consists of `Var` objects and differentiable `Function` objects.

A forward pass constructs a computational graph:

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

Each resulting variable keeps track of the function that produced it.

Calling:

```python
y.backward()
```

traverses the graph in reverse order and propagates gradients through each operation.

The engine also tracks function levels so the graph can be traversed in the correct order.

## Operations

Current work includes operations such as:

```text
Add
Sub
Mul
Div
Neg
Pow
Exp
Sin
Sum
Reshape
Transpose
MatMul
```

Broadcasted binary operations reduce their gradients back to the original input dimensions when necessary.

## Gradient Checking

Chris Torch includes numerical gradient checking for validating autograd implementations.

The analytical gradient produced by the computational graph can be compared against a numerical approximation based on finite differences.

Conceptually:

```text
Analytical gradient
        │
        │ compare
        ▼
Numerical gradient
```

This is useful when implementing new differentiable operations.

## AdaptiveCpp Backend

Chris Torch includes an experimental native backend using C++, SYCL, and AdaptiveCpp.

[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) is a heterogeneous C++ compiler/runtime ecosystem supporting SYCL applications across different hardware backends.

The current integration follows this architecture:

```text
Chris Torch
    │
    ▼
Python
    │
    ▼
Native binding
    │
    ▼
libchristorch_acpp.so
    │
    ▼
C++ / SYCL
    │
    ▼
AdaptiveCpp
    │
    ▼
Execution backend
```

The native backend is compiled into:

```text
build/libchristorch_acpp.so
```

and exposed to Python through a native interface.

## Native MatMul

Matrix multiplication is the first operation being used to experiment with the AdaptiveCpp backend.

For:

```text
A[M × K]
B[K × N]
```

the kernel computes:

```text
C[M × N] = A × B
```

The initial implementation uses a two-dimensional SYCL execution range.

Each work-item computes one output element:

```cpp
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
```

This is intentionally a naive implementation.

Its primary purposes are:

* validating the native backend;
* testing Python/native integration;
* understanding SYCL execution;
* establishing a performance baseline;
* providing a starting point for optimization.

## Python ↔ Native Integration

The native implementation exposes a C-compatible interface that can be loaded from Python.

Conceptually:

```text
MatMul.forward()
       │
       ▼
AdaptiveCpp Python backend
       │
       ▼
ctypes
       │
       ▼
ct_matmul_f32()
       │
       ▼
SYCL MatMul
```

This makes it possible to progressively replace operations such as:

```python
np.dot(x, W)
```

with native implementations.

The same MatMul kernel can also be used by autograd during the backward pass:

```text
                gy
               /  \
              /    \
             ▼      ▼
        gy × Wᵀ    xᵀ × gy
             │      │
             └──┬───┘
                ▼
          Native MatMul
```

## Native Backend Structure

The native portion of the repository follows a simple structure:

```text
chris-torch/
├── native/
│   ├── matmul.hpp
│   ├── matmul.cpp
│   └── ...
│
├── benchmarks/
│   ├── bench_matmul.cpp
│   └── ...
│
├── build/
│   ├── libchristorch_acpp.so
│   │
│   └── benchmarks/
│       └── bench_matmul
│
└── Makefile
```

Additional native operations can be added under `native/`.

The Makefile automatically discovers native `.cpp` files, allowing the backend to grow without maintaining a manual source list.

## Building the AdaptiveCpp Backend

AdaptiveCpp must be installed before building the native backend.

The current development environment uses:

```text
/etc/acpp/bin/acpp
```

Build:

```bash
make
```

This generates:

```text
build/libchristorch_acpp.so
```

Clean:

```bash
make clean
```

Rebuild:

```bash
make rebuild
```

## Benchmarking

Native kernels are benchmarked independently from Python and the autograd engine.

Run:

```bash
make benchmark
```

### Current MatMul Baseline

Current development environment:

```text
Device: AdaptiveCpp OpenMP host device
Kernel: naive SYCL MatMul
Matrix: 1024 × 1024 × 1024
Runs:   20
```

Current measured baseline:

```text
Average:     ~362.7 ms
Performance: ~5.92 GFLOPS
```

This number represents the current naive implementation and serves only as a baseline for future optimization.

It should not be interpreted as the performance limit of AdaptiveCpp, SYCL, or the hardware.

## MatMul Optimization Roadmap

The current kernel intentionally starts from the simplest implementation.

Planned experiments can progressively explore:

```text
Naive MatMul
     │
     ▼
nd_range
     │
     ▼
Work-groups
     │
     ▼
Tiling
     │
     ▼
Local Memory
     │
     ▼
Vectorization
     │
     ▼
Memory Access Optimization
     │
     ▼
Backend-specific Optimization
```

Performance improvements can then be measured against the original ~5.92 GFLOPS baseline.

## Memory Model

The first native implementation uses a simple interoperability model.

Currently:

```text
NumPy memory
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
NumPy memory
```

This prioritizes simplicity while the backend architecture is being developed.

It is not the intended final memory architecture.

A more advanced implementation could introduce a native tensor abstraction:

```text
Tensor
  │
  ├── shape
  ├── strides
  ├── dtype
  ├── device
  │
  └── data
       │
       ▼
   Native memory
       │
       ▼
   SYCL kernels
```

This would allow tensors to remain on the target device across multiple operations.

For example:

```text
Host
 │
 │ initial transfer
 ▼
GPU Tensor
 │
 ├── MatMul
 │
 ├── Add
 │
 ├── ReLU
 │
 ├── MatMul
 │
 └── Softmax
 │
 ▼
Host
```

instead of copying between Python and the device after every operation.

## Planned Native Kernels

Potential operations for the AdaptiveCpp backend include:

```text
Tensor operations
├── add
├── sub
├── mul
├── div
└── reductions

Linear algebra
├── matmul
├── transpose
└── batched matmul

Activations
├── ReLU
├── GELU
└── SiLU

Neural network operations
├── Softmax
├── normalization
└── embedding operations
```

Implementations will be added progressively as the backend evolves.

## Neural Networks

Chris Torch is intended to progressively explore higher-level neural network components on top of the autograd engine.

Areas of interest include:

```text
MLP
CNN
RNN
Transformers
```

The goal is not only to implement the models, but also to understand how their operations eventually map down to tensor kernels and hardware execution.

## Relationship with Chris GPT

Chris GPT is intended to explore GPT-style model training using the same general ecosystem.

Conceptually:

```text
Chris GPT
    │
    ▼
Transformer
    │
    ▼
Chris Torch
    │
    ├── Autograd
    ├── Tensor Operations
    │
    └── Native Backend
             │
             ▼
          SYCL
             │
             ▼
        AdaptiveCpp
```

This makes Chris Torch the framework-level component of the broader experimentation stack.

## Relationship with Chris Llama

[Chris Llama](https://github.com/christianrss/chris-llama) explores the other side of the model lifecycle: inference.

Conceptually:

```text
              Model
             /     \
            /       \
           ▼         ▼
      Training     Inference
          │            │
          ▼            ▼
     Chris Torch   Chris Llama
          │            │
          ▼            ▼
     Chris GPT       C runtime
```

The projects intentionally explore different layers rather than relying entirely on existing ML frameworks and inference engines.

## Relationship with AdaptiveCpp

The AdaptiveCpp backend also provides a practical environment for learning heterogeneous C++ and exploring the compiler/runtime stack underneath SYCL applications.

Related repositories:

* [AdaptiveCpp upstream](https://github.com/AdaptiveCpp/AdaptiveCpp)
* [My AdaptiveCpp fork](https://github.com/christianrss/AdaptiveCpp)

The separate `tests-acpp` repository is used for smaller isolated AdaptiveCpp/SYCL experiments and compiler/runtime testing before concepts are integrated into Chris Torch.

## Development Roadmap

The general direction of the project is:

```text
Autograd
   │
   ▼
Tensor Operations
   │
   ▼
Neural Network Primitives
   │
   ▼
Native Kernels
   │
   ▼
Tensor / Device Abstraction
   │
   ▼
Kernel Optimization
   │
   ▼
Heterogeneous Execution
   │
   ▼
Model Training Experiments
```

Near-term areas of experimentation include:

* expanding autograd operations;
* improving gradient validation;
* integrating native MatMul with the Python API;
* adding additional SYCL kernels;
* improving the native memory model;
* experimenting with optimized MatMul implementations;
* comparing CPU and GPU execution;
* benchmarking kernel implementations;
* developing the tensor/device abstraction required to avoid unnecessary memory transfers.

## Development Philosophy

Chris Torch deliberately implements fundamental components instead of hiding them behind an existing deep learning framework.

The project is primarily about understanding:

* how automatic differentiation actually works;
* how computational graphs are constructed and traversed;
* how tensor broadcasting affects gradient computation;
* how matrix operations are implemented;
* how Python interacts with native C++;
* how kernels map onto parallel hardware;
* how memory movement affects performance;
* how heterogeneous runtimes execute kernels;
* how low-level optimization affects machine learning workloads.

Correctness comes before optimization.

Simple implementations establish baselines before more advanced implementations are introduced.

## Status

Chris Torch is under active development.

Currently implemented or under active experimentation:

```text
✓ Python/NumPy autograd engine
✓ Computational graph
✓ Reverse-mode differentiation
✓ Broadcasting-aware gradients
✓ Gradient checking
✓ Core differentiable operations
✓ Matrix multiplication
✓ Experimental C++/SYCL backend
✓ AdaptiveCpp integration
✓ Native shared library
✓ Native MatMul kernel
✓ Native MatMul benchmark
✓ AdaptiveCpp OpenMP host execution

○ Additional native kernels
○ Optimized MatMul
○ Native tensor abstraction
○ Persistent device memory
○ GPU backend testing
○ MLP
○ CNN
○ RNN
○ Transformer experiments
○ Chris GPT integration
```

## References

* [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)
* [AdaptiveCpp Documentation](https://adaptivecpp.github.io/)
* [Khronos SYCL](https://www.khronos.org/sycl/)
* [SYCL Registry](https://registry.khronos.org/SYCL/)
* [Chris Llama](https://github.com/christianrss/chris-llama)
* [My AdaptiveCpp Fork](https://github.com/christianrss/AdaptiveCpp)

## License

Chris Torch is an experimental project intended for educational, research, and systems programming exploration.

Third-party projects referenced or used by Chris Torch are distributed under their respective licenses.
