# num-clj (数)

[![CI](https://github.com/kotoba-lang/num/actions/workflows/ci.yml/badge.svg)](https://github.com/kotoba-lang/num/actions/workflows/ci.yml)

A **GPU-accelerated numerical compute library** for Clojure — BLAS-level dense and
sparse linear algebra (the NumPy/cuBLAS layer) with a **portable contract + injected
backends** so the *same* program runs on a pure-Clojure CPU, a WebGPU compute pipeline
in a browser tab, or cuBLAS on an H100. The core (array type, op protocol, CPU reference
backend, public API) is **zero-dependency, all `.cljc`** — it runs on JVM / SCI /
ClojureScript / GraalVM / kotoba-WASM. GPU backends carry the native code and are
**host-injected** behind one protocol; the core never pulls a toolchain.

Sibling of [kudaki-clj](https://github.com/com-junkawasaki/kudaki-clj) (explicit FEA) and
[nagare-clj](https://github.com/com-junkawasaki/nagare-clj) (finite-volume CFD): those
zero-dep solvers express their hot loops (Krylov `SpMV`/`AXPY`, element `GEMM`) and, by
**injecting a num-clj GPU backend**, run on the GPU *without changing the portable
kernel* — the same way the design actor swaps a Store or an Advisor. Fits the wgpu/WebGPU
line of [kami-webgpu](https://github.com/com-junkawasaki/kami-webgpu) (declarative render)
and [kami-engine](https://github.com/com-junkawasaki/kami-engine) — num-clj is the GPU
**compute** half they lacked.

## Why a contract + injected backends (the whole idea)

One seam — `num.protocol/IBackend` — is all that separates "where the math runs" from
"what the math is". A `handle` is an opaque buffer: a `double-array` on CPU, a `GPUBuffer`
on WebGPU, a device pointer on CUDA. The array layer only moves handles through the
protocol, so:

```
num.core/matmul, spmv, axpy! …        ← one portable program (your code)
        │  dispatches through
        ▼
num.protocol/IBackend                 ← the ONE seam
        ├── num.cpu     (pure .cljc, zero-dep)      the ORACLE / fallback
        ├── num.wgsl    (WGSL compute via wgpu)     PORTABLE PRIMARY
        ├── :cuda       (cuBLAS / cuSPARSE)         NVIDIA fast path
        ├── :metal      (Metal Performance Shaders) Apple fast path
        └── :rocm       (rocBLAS / hipSPARSE)       AMD fast path
```

This is the same injection pattern as `MemStore ‖ DatomicStore` and the `cae-solver`
`[:solver :kind]` dispatch used across com-junkawasaki.

## "Optimized for each GPU" — without three vendor kernels

The **primary** GPU backend is **WGSL compute via WebGPU/wgpu**. You write a compute
shader once (`num.wgsl`), and wgpu compiles it to each GPU's native ISA:

| Target | wgpu lowers WGSL to | Reaches |
|---|---|---|
| Apple | **Metal MSL** | M-series / AMD on macOS |
| Vulkan | **SPIR-V** | AMD (ROCm-class), NVIDIA, Intel, Linux |
| D3D12 | **DXIL** | Windows NVIDIA/AMD/Intel |
| Browser | **WebGPU** | any GPU, in a tab |

So one shader source is **per-GPU-optimized by the driver**. The shipped kernels
(`num.wgsl`) include a **16×16 shared-memory-tiled GEMM** (the canonical optimized GPU
kernel, ~16× less global traffic), a **scalar-CSR SpMV**, **AXPY**, and a **tree
reduction**. When raw vendor throughput matters, the optional `:cuda` / `:metal` /
`:rocm` backends drop cuBLAS / MPS / rocBLAS in behind the *same* `IBackend` — opt-in,
JVM/native only, never required.

## API (`num.core`, on `num.array` values)

```clojure
(require '[num.cpu :as cpu] '[num.array :as a] '[num.core :as nm] '[num.sparse :as sp])
(def b (cpu/cpu-backend))               ; swap for a GPU backend; nothing else changes

(def A (a/from-vec b [1 2 3 4] [2 2]))  ; 2×2
(def x (a/from-vec b [1 1] [2]))
(a/->vec (nm/matvec A x))               ;=> [3.0 7.0]
(nm/dot x x)                            ;=> 2.0
(a/->vec (nm/matmul A A))               ;=> [7.0 10.0 15.0 22.0]

(def K (sp/dense->csr 2 3 [1 0 2  0 3 0]))
(a/->vec (nm/spmv b K x))               ;=> [3.0 3.0]   (sparse A·x)
```

Ops: `axpy! scal! dot nrm2 add sub mul div sum amax amin matvec matmul spmv`.

## N-D tensors (`num.tensor`, ADR-2607051400 §Phase 1)

`num.core` above is deliberately 1-D-vector/2-D-matrix only (BLAS-shaped). `num.tensor`
sits on top and adds a real, arbitrary-rank, shape-aware layer: broadcasting, shape
manipulation, axis reductions, and batched matmul. It **extends `num.array/NDArray`**
rather than introducing a new type — `NDArray` is already `{:backend :handle :shape}`
with an arbitrary-length `shape` vector, and `nelems`/`from-vec`/`->vec` already work for
rank 0 (scalar) through arbitrary rank with no changes; what was missing was the
*operations*, not the type. `num.core` itself is untouched — same fast, non-broadcasting,
direct-to-`IBackend` dispatch as before, zero behavior change.

```clojure
(require '[num.cpu :as cpu] '[num.array :as a] '[num.tensor :as t])
(def b (cpu/cpu-backend))

;; broadcasting (NumPy-style: align from the trailing dim, size-1 stretches)
(def col (a/from-vec b [1 2 3] [3 1]))
(def row (a/from-vec b [10 20 30 40] [1 4]))
(a/->vec (t/add col row))                 ;=> [3 4] outer-sum, row i + col j per cell
(t/broadcast-shapes [3 4] [2 1 4])        ;=> [2 3 4]
(t/broadcast-shapes [2 3] [2 4])          ;=> throws (incompatible, non-1 mismatch)

;; reshape / transpose / squeeze / unsqueeze (data-preserving)
(def m (a/from-vec b (range 1 7) [2 3]))  ; [[1 2 3][4 5 6]]
(:shape (t/reshape m [3 2]))              ;=> [3 2]              (zero-copy relabel)
(a/->vec (t/transpose m))                 ;=> [1 4 2 5 3 6]      ([3 2], real data move)
(:shape (t/squeeze (a/from-vec b [1] [1 1 1])))    ;=> []
(:shape (t/unsqueeze m 0))                ;=> [1 2 3]

;; axis-parameterized reductions (nil axis = reduce everything; NumPy keepdims convention)
(a/->vec (t/sum m 0))                     ;=> [5 7 9]            (down columns)
(a/->vec (t/sum m 1))                     ;=> [6 15]             (across rows)
(:shape (t/sum m 1 {:keepdims? true}))    ;=> [2 1]
(a/->vec (t/mean m 0))                    ;=> [2.5 3.5 4.5]
(a/->scalar (t/amax m))                   ;=> 6.0                (full reduction)

;; batched matmul: last two dims are the matrix dims, leading dims broadcast
(def A (a/from-vec b (range 1 13) [2 2 3]))   ; 2 batches of 2×3
(def B (a/from-vec b (range 1 13) [2 3 2]))   ; 2 batches of 3×2
(:shape (t/matmul A B))                   ;=> [2 2 2]
```

**Host-materialized, not device-native (an explicit, documented tradeoff):**
`num.protocol/IBackend` has no notion of strides/gather/scatter — a handle is an opaque
flat contiguous buffer. So `broadcast-to`/`transpose`/axis-reductions/`matmul` here read
operands back via `arr/->vec`, compute with plain `double-array` loops (same style as
`num.cpu`'s reference loops), and re-upload via `arr/from-vec` — correct on ANY backend
today, but a host round-trip. `reshape`/`squeeze`/`unsqueeze` are the exception: for a
row-major contiguous layout they never move data, so they're zero-copy metadata edits.
Pushing the round-tripping ops into `IBackend` itself (so a GPU backend executes them
without leaving the device) is **Phase 2** work per ADR-2607051400, not this pass.

**Phase-1 scope vs. what's deferred, honestly:**
- Done: arbitrary-rank shape (the shape math — `row-major-strides`/`unravel`/`ravel`/
  `broadcast-shapes` — is plain `(count shape)`-generic with no rank ceiling anywhere;
  explicitly TESTED, not just asserted generic, at 0-D/1-D/2-D/3-D and at rank 4 per the
  ADR's "at least 4-D": `rank-4-reshape-and-transpose`, `rank-4-axis-reduction-matches-naive`,
  and `rank-4-batched-matmul-matches-naive` exercise reshape/transpose/axis-sum/batched-matmul
  on 4-D tensors against independently-written naive references), NumPy-style
  broadcasting (`broadcast-shapes`/`broadcast-to`) with the exact trailing-align/size-1
  rule, `reshape`/`transpose`/`squeeze`/`unsqueeze` (data-preserving, tested against
  hand-computed layouts), axis-parameterized `sum`/`amax`/`amin`/`mean` with `:keepdims?`
  (single axis, multiple axes, or all axes), and batched N-D `matmul` with batch-dim
  broadcasting.
- Deferred (not attempted this pass, left to Phase 2 per the ADR): a live GPU dispatch
  path for any of the above (everything here runs through the CPU-materialized
  `->vec`/`from-vec` seam regardless of which `IBackend` is injected); new WGSL kernels
  for N-D broadcast/batched-matmul (the ADR explicitly calls this out as Phase 2 net-new
  shader work); and proving `num.tensor` under ClojureScript via the `cljs-verify`
  harness (it's written in the same portable `.cljc` style — `Math/…`, `double-array`,
  `aget`/`aset`, `ex-info` — as the rest of this repo, but `test/num/cljs_verify.cljs`
  currently only exercises `num.contract` against `num.cpu`, not `num.tensor`; that
  wiring wasn't extended in this pass).

## Live GPU backend (Deno + WebGPU → Metal, ADR-2607051400 §Phase 2)

`num.deno-gpu` promotes `verify/metal_contract.js`'s raw JS harness into a REAL
`num.wgsl/IGpuDevice` + `num.protocol/IBackend` implementation — `num.core`'s ops
(`axpy!`/`scal!`/`add`/`sub`/`mul`/`div`/`sum`/`amax`/`amin`/`dot`/`nrm2`/`matvec`/
`matmul`/`spmv`) now dispatch through the SAME WGSL kernels on real GPU hardware from
real Clojure code (ClojureScript compiled for Deno), not only from a standalone script.

```clojure
(require '[num.array :as a] '[num.core :as nm] '[num.deno-gpu :as dg])
(-> (dg/gpu-backend)                              ; async: device negotiation is Promise-based
    (.then (fn [gpu]
             (let [x (a/from-vec gpu [1 2 3] [3])
                   y (a/from-vec gpu [4 5 6] [3])]
               (.then (nm/dot x y) println)))))   ; host-value ops (dot/nrm2/sum/amax/amin,
                                                   ; arr/->vec) return a JS Promise on this
                                                   ; backend — see num.deno-gpu docstring
```

**Sync vs. async, resolved (not hand-waved):** a single WebGPU queue processes
submitted commands strictly in order, so `-alloc`/`-copy-from-host`/`-axpy`/`-scal`/
`-ewise`/`-gemv`/`-gemm`/`-spmv` are fully synchronous end-to-end — dispatching
`axpy!`/`scal!`/`add`/`sub`/`mul`/`div`/`matvec`/`matmul`/`spmv` through the Deno GPU
backend behaves exactly like a fully-synchronous backend would. Only the four ops that
read a value back to the host (`-copy-to-host`/`-reduce`/`-dot`/`-nrm2` — i.e.
`arr/->vec`, `sum`, `amax`, `amin`, `dot`, `nrm2`) are unavoidably async (`GPUBuffer.
mapAsync`) and return a JS `Promise` instead of an immediate value — a deliberate,
documented deviation from `IBackend`'s normal contract for this one host. Native
ClojureScript `async`/`await` (`^:async`/`js-await`) was tried first and confirmed NOT
supported by this repo's plain `clojure -M -m cljs.main` pipeline (no shadow-cljs); the
backend uses plain JS Promise `.then` interop instead, the same style
`kotoba-lang/host`'s `kami.backend.browser` already uses for its own browser GPU host
bring-up.

**Run it for real** (the only host with GPU access — confirmed working with Deno ≥ 2,
no `--unstable-webgpu` flag needed):

```bash
clojure -M:deno-verify && deno run --allow-all target/deno-gpu-verify.cjs
```

This cross-checks the live GPU backend against `num.cpu`'s reference oracle (dispatched
through `num.core`/`num.array`, i.e. through `IBackend`, the same seam any real caller
uses) — **verified passing on real Apple M1 Max Metal hardware while building this**:
`Deno WgslBackendAsync ≡ CPU oracle: 14 passed, 0 failed` (all 14 `num.core` ops:
axpy!/scal!/add/sub/mul/div/dot/nrm2/sum/amax/amin/matvec/matmul/spmv).

## Status

S0 is real and tested: the array/CSR types, the `IBackend` protocol, the **pure-Clojure
CPU reference backend**, the public API, and the **backend contract suite** (every op
checked against reference values — `CpuBackend ≡` any future GPU backend).

**S1 (live WGSL backend) is now real, not just shipped shaders.** `num.wgsl` ships the
compute shaders + the `IGpuDevice` host port + the dispatch plan; `num.wgsl-backend`
dispatches them through an injected device (synchronous variant, for a future native/
blocking device — not yet implemented); and **`num.deno-gpu` is a live, working
`IGpuDevice` + `IBackend` for Deno's native WebGPU**, verified end-to-end against the CPU
oracle through `num.core` on real Apple Metal hardware (previous section).

**S2 CUDA dispatch is now implemented, but NVIDIA qualification is not yet
claimed.** `num.cuda/CudaBackend` implements every `IBackend` operation through
an injected `ICudaDriver`: CUDA allocation/transfer, cuBLAS level 1/2/3,
cuSPARSE CSR SpMV and CUDA elementwise/reduction kernels. Handles enforce owner,
size and explicit-free lifecycle; device/runtime/cuBLAS/cuSPARSE provenance is
mandatory. The complete fake-driver contract passes, proving dispatch and
lifecycle behavior without requiring CUDA on every developer machine. A native
NVIDIA host adapter and real-GPU contract run remain required before performance
or hardware qualification claims; see `docs/adr/0002-cuda-native-backend.md`.
ROCm and JVM Panama/wgpu-native remain unimplemented. The Metal native fast path
is implemented and hardware-qualified below.

The native side is implemented under `native/cuda`: a stable C ABI backed by
CUDA Runtime, cuBLAS, cuSPARSE, Thrust reductions and a custom elementwise
kernel. `scripts/verify-cuda.sh` builds it and runs allocation/copy, reduction,
GEMM and CSR SpMV smoke checks, then loads the shared library through the
optional JNA adapter and runs all 14 `num.contract` operations; the manual
`cuda-native.yml` workflow requires
an explicitly labelled NVIDIA self-hosted runner and fails when CUDA is absent.
This repository's current Apple host cannot supply the final NVIDIA evidence.

`num.solver/pcg` is the first production consumer of the complete backend
contract. CSR SpMV and vector updates stay resident on the selected backend;
only dot-product convergence scalars return to the host. It explicitly releases
temporary device buffers, reports iterations/residual/backend provenance and
fails on non-positive-definite systems. The same solver passes on the CPU oracle
and injected CUDA driver, where tests confirm cuSPARSE SpMV plus cuBLAS
SDOT/SAXPY dispatch rather than a hidden CPU solve.

`num.gpu-compiler` now depends on a pinned `kotoba-lang/compiler` accelerator
backend. Seven numerical kernel specifications produce 14 independently
verified artifacts: WGSL and CUDA for add/sub/mul/div and sum/max/min. Both
targets share the same typed KIR SHA-256 while retaining distinct code hashes.
CUDA backend provenance embeds every generated KIR/code hash, allowing CAE
qualification to identify the exact compiler output rather than merely saying
"CUDA". The compiler has no dependency back to `num`.
The native JNA driver sends those verified CUDA sources to NVRTC, loads PTX via
the CUDA Driver API, caches seven functions and launches generated ewise/reduce
kernels. Its contract asserts seven NVRTC compilations, compiled-kernel
execution and zero bootstrap ewise/reduction calls.

Native Metal now uses the same architecture. `num.metal/MetalBackend` and the
Objective-C++ C ABI under `native/metal` implement owned `MTLBuffer` lifecycle,
vector operations, dense GEMV/GEMM, CSR SpMV, pipeline caching, and runtime
compilation of compiler-generated MSL. The optional JNA adapter keeps native
dependencies outside the default graph and records Apple device, OS, runtime,
and all seven KIR/code hashes. `scripts/verify-metal-native.sh` passes the full
14-operation `IBackend` contract and the independent generated-MSL gate on an
Apple M1 Max. These dense kernels are direct Metal compute kernels; using Metal
Performance Shaders as an additional tuned implementation remains future work.

```bash
clojure -M:test                                  # CPU + injected CUDA dispatch contracts (JVM)
clojure -M:cljs && node target/cljs-verify.js     # PROOF: the same .cljc core runs under ClojureScript
clojure -M:deno-verify && deno run --allow-all target/deno-gpu-verify.cjs   # PROOF: live GPU ≡ CPU oracle, real Metal
./scripts/verify-metal-native.sh                   # PROOF: native Metal 14/14 + compiler MSL gate
```

**Portability is proven, not just claimed.** The `.cljc` core (CPU backend, array/CSR
types, the full op contract) compiles to JS via ClojureScript and runs green on node —
`cljs CPU contract: 14 passed, 0 failed`. The WGSL kernels are verified on real
**Apple M4/M1 Metal** three separate ways now: the standalone harness
(`verify/metal_contract.js`, 13/13) including a Jacobi-PCG Poisson solve
(`verify/metal_pcg.js`), AND the live `num.deno-gpu` backend dispatched through real
`num.core` Clojure code (`deno-gpu-verify`, 14/14 against the CPU oracle). So num-clj
genuinely spans pure-Clojure → cljs → live GPU from one source, with the GPU path no
longer only exercised by a script outside the Clojure dispatch seam.

**Honest gap:** `num.tensor`'s N-D ops (broadcast/reshape/axis-reduce/batched-matmul,
ADR-2607051400 §Phase 1) do not yet dispatch through `num.deno-gpu` or any GPU backend —
they are host-materialized (round-trip through `arr/->vec`/`arr/from-vec`) regardless of
which `IBackend` is injected. Extending the WGSL kernel set to N-D broadcast/batched-
matmul dispatch is unimplemented net-new shader work, not attempted in this pass.
