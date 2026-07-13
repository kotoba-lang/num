# ADR-0004 — Native Metal compiler backend

- Status: accepted; complete native contract hardware-qualified
- Date: 2026-07-12

`MetalBackend` mirrors CUDA ownership and numerical semantics behind an injected
`IMetalDriver`. The host port covers Metal buffers, BLAS-style vector operations,
dense matrix operations, CSR SpMV and compiled elementwise/reduction kernels.
Handles reject cross-backend access and use after explicit free.

`num.gpu-compiler` requests `:msl-v1` artifacts from the pinned Kotoba compiler.
Metal provenance binds device, GPU family, OS, compiler version and all seven
KIR/code hashes. A compiled driver runtime-compiles MSL and caches pipeline
functions; bootstrap kernels are not selected when it is available.

The native Objective-C++ bridge exposes a stable C ABI, owns shared `MTLBuffer`
allocations, synchronously checks command-buffer failures, and caches bootstrap
and generated compute pipelines. JNA is opt-in through `:metal-jna`, preserving
the dependency-free portable core. Compiler artifacts are recompiled through
`MTLDevice.newLibraryWithSource` and selected for all ewise/reduction dispatch.

`scripts/verify-metal-native.sh` builds the dylib and passes all 14 operations
(DOT, NRM2, AXPY, SCAL, four ewise, three reductions, GEMV, GEMM, CSR SpMV)
against the CPU oracle on an Apple M1 Max. It then runs an independent gate for
compiler-generated add and sum MSL. Dense GEMV/GEMM use
`MPSMatrixMultiplication` (including GEMV as matrix × single-column matrix),
with direct Metal compute kernels retained as a fallback implementation. Runtime
provenance reports `MetalPerformanceShaders` as the selected dense provider.
