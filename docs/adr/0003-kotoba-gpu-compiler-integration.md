# ADR-0003 — kotoba-lang/compiler owns generated numerical GPU kernels

- Status: accepted; elementwise/reduction artifacts integrated
- Date: 2026-07-12

`num` depends on a pinned `kotoba-lang/compiler` revision. It describes each
f32 elementwise/reduction operator as typed accelerator KIR and requests sealed
WGSL and CUDA artifacts. The compiler does not depend on `num`.

CUDA backend provenance records the compiler target plus KIR/code hashes for
all seven generated kernels. Artifact generation and verification fail before
native device work if operator semantics, resource bounds, seal or regenerated
code differ. Runtime buffers and cuBLAS/cuSPARSE remain owned by `num`.

The current native bridge still contains bootstrap elementwise/reduction
implementations. A subsequent NVRTC loader may consume the verified CUDA source
directly, but it must preserve these hashes in qualification evidence. WGSL
pipeline replacement follows the same registry; cross-target tests already
require identical KIR identity and distinct target code identity.
