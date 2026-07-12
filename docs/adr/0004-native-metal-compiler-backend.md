# ADR-0004 — Native Metal compiler backend

- Status: accepted; backend contract and generated-kernel hardware gate implemented
- Date: 2026-07-12

`MetalBackend` mirrors CUDA ownership and numerical semantics behind an injected
`IMetalDriver`. The host port covers Metal buffers, BLAS-style vector operations,
MPS dense matrix operations, CSR SpMV and compiled elementwise/reduction kernels.
Handles reject cross-backend access and use after explicit free.

`num.gpu-compiler` requests `:msl-v1` artifacts from the pinned Kotoba compiler.
Metal provenance binds device, GPU family, OS, compiler version and all seven
KIR/code hashes. A compiled driver runtime-compiles MSL and caches pipeline
functions; bootstrap kernels are not selected when it is available.

The hardware gate passes compiler-generated ewise-add and sum-reduction kernels
through `MTLDevice.makeLibrary(source:)` on an Apple M1 Max, executes both on the
GPU and checks exact expected f32 results. The existing WGSL backend separately
passes the complete IBackend contract through WebGPU→Metal. A native Objective-C
or Swift `IMetalDriver` implementation of every BLAS/MPS method remains required
before claiming the complete native-Metal contract on hardware.
