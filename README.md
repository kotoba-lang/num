# num-clj (数)

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

## Status

S0 is real and tested: the array/CSR types, the `IBackend` protocol, the **pure-Clojure
CPU reference backend**, the public API, and the **backend contract suite** (every op
checked against reference values — `CpuBackend ≡` any future GPU backend). The WGSL
backend ships its compute **shaders + the `IGpuDevice` host port + the dispatch plan**;
wiring a live device (browser WebGPU or a native wgpu host) and the vendor-native
backends are the next stages — see `docs/adr/0001-architecture.md`.

```bash
clojure -X:test     # CPU backend satisfies the full IBackend contract
```
