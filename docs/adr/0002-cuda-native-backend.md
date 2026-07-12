# ADR-0002 — CUDA native backend and CAE integration

- Status: accepted; portable backend implemented, native NVIDIA conformance pending
- Date: 2026-07-12

## Context

`num` already has one portable `IBackend` and a live WGSL/WebGPU implementation.
CUDA was previously only a row in ADR-0001. The CAD/CAE suite needs sparse
iterative solves and dense geometry kernels on NVIDIA workstations without
making CUDA a transitive dependency of CLJC applications.

## Decision

`num.cuda/CudaBackend` implements the complete `IBackend`. It owns opaque
`CudaHandle` values and delegates native work to an injected `ICudaDriver`.
The driver boundary covers:

- CUDA allocation, explicit free, f32 host/device transfer;
- cuBLAS SAXPY, SSCAL, SDOT, SNRM2, SGEMV and SGEMM;
- cuSPARSE CSR SpMV;
- CUDA elementwise and reduction kernels.

Handles record their backend owner, allocation length and released state.
Foreign, undersized and use-after-free handles fail before a native call.
Every driver must report device, CUDA runtime, cuBLAS and cuSPARSE versions;
qualification artifacts retain this map.

Dense data remains row-major. A cuBLAS host adapter performs the conventional
operand/transpose swap internally; callers and the portable contract never see
column-major storage. CSR indices are zero-based `int32`, values are `float32`.

CUDA is opt-in and JVM/NVIDIA-only. `src/num/cuda.clj` has no CUDA dependency;
native bindings live in a separately loaded host package/alias. Absence of the
CUDA library is a capability result, never a namespace load failure. No silent
CPU fallback is permitted after a CUDA backend has been selected.

## Verification gates

1. The injected-driver suite must pass all 14 `num.contract` operations and
   lifecycle/provenance negative tests on every platform.
2. NVIDIA release CI must run the same contract on a real device and record
   GPU name, compute capability, driver/runtime/cuBLAS/cuSPARSE versions.
3. Numerical acceptance is CPU-oracle agreement within f32 tolerance.
4. Performance gates compare warmed device-resident operations, excluding
   allocation and transfer, and publish matrix/CSR dimensions and iteration
   counts. A speedup claim without those fields is invalid.
5. `kami-engine-modeling` CAE selects the backend through its solver adapter;
   studies and results retain backend provenance. CPU and CUDA solutions must
   satisfy the same reaction/energy/convergence gates.

## Initial kernel policy

- BLAS and SpMV use vendor libraries rather than handwritten replacements.
- Fused elementwise/reduction kernels are allowed only behind `ICudaDriver`.
- FP32 is the first accelerated contract because WGSL and broad GPUs share it.
  FP64 becomes a separate capability and qualification tier; it must not be
  inferred from an NVIDIA device name.
- Sparse matrix structure may be cached by content revision in the native
  driver, but a cache cannot change observable ownership or result ordering.

## Consequences

- Portable consumers can inject CUDA without changing numerical algorithms.
- CUDA failures are explicit and attributable to a driver call and device.
- NVIDIA deployment requires a separately installed native host adapter and
  cannot be claimed from the fake-driver contract alone.
