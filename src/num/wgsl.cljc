(ns num.wgsl
  "The WGSL/WebGPU backend SPEC — the portable-primary GPU path.

  One WGSL compute shader, written once, is compiled by wgpu/WebGPU to each GPU's
  NATIVE ISA: **Metal MSL** on Apple, **SPIR-V** on Vulkan (AMD/ROCm-class and
  NVIDIA), **DXIL** on D3D12, and runs in a browser tab unchanged. That is how
  num-clj reaches 'optimized for each GPU' without writing three vendor kernels —
  the driver does the per-GPU codegen and scheduling.

  num-clj ships NO native code. Execution is HOST-INJECTED through `IGpuDevice`
  (the same port-injection pattern as the actors' Store/Advisor): a browser host
  implements it over the `navigator.gpu` WebGPU API from ClojureScript; a JVM/
  native host implements it over a wgpu binding. A `WgslBackend` (host-side)
  satisfies `num.protocol/IBackend` by binding buffers and dispatching the shaders
  below, falling back to a host read-compute-write for any op it has not yet
  accelerated — so it is always a COMPLETE backend, GPU-fast on the hot kernels.

  The shaders here are the real, reviewable artifacts; the contract test
  (`num.contract`) is what proves a live WgslBackend ≡ the CPU reference."
  )

;; ---------------------------------------------------------------------------
;; Host port — the seam the browser / native host fills in (no native code here)
;; ---------------------------------------------------------------------------

(defprotocol IGpuDevice
  "WebGPU/wgpu primitives provided by the host. Buffers are f32 storage unless
  noted; `usage` ∈ #{:storage :uniform :read}. Pure data in, handles out."
  (-create-buffer [dev n usage] "Allocate an n-element device buffer → buf handle.")
  (-write-buffer [dev buf xs] "Upload host seq `xs` into `buf`.")
  (-read-buffer [dev buf n] "Download `n` elements from `buf` → host vector.")
  (-compile [dev wgsl-src entry] "Compile a WGSL module's `entry` fn → pipeline.")
  (-dispatch [dev pipeline buffers workgroups]
    "Run `pipeline` with `buffers` bound at @binding 0.. and `workgroups`
     [x y z] workgroup counts."))

;; ---------------------------------------------------------------------------
;; Compute shaders (WGSL) — compiled per-GPU by wgpu
;; ---------------------------------------------------------------------------

(def axpy-wgsl
  "Level-1 AXPY: y ← αx + y. One invocation per element, 64-wide workgroups."
  "
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform>             alpha: f32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&y)) { return; }
  y[i] = alpha * x[i] + y[i];
}")

(def spmv-csr-wgsl
  "Sparse mat-vec y = A·x, A in CSR. One thread per row accumulates its nonzeros —
  the standard scalar-CSR kernel (coalescing-friendly enough for the
  verification-scale systems nagare/kudaki Krylov solvers throw at it)."
  "
@group(0) @binding(0) var<storage, read>       row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read>       col_idx: array<u32>;
@group(0) @binding(2) var<storage, read>       vals:    array<f32>;
@group(0) @binding(3) var<storage, read>       x:       array<f32>;
@group(0) @binding(4) var<storage, read_write> y:       array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&y)) { return; }
  var s: f32 = 0.0;
  let a = row_ptr[i];
  let b = row_ptr[i + 1u];
  for (var p: u32 = a; p < b; p = p + 1u) {
    s = s + vals[p] * x[col_idx[p]];
  }
  y[i] = s;
}")

(def gemm-tiled-wgsl
  "Dense GEMM C = A·B (A m×k, B k×n, row-major) with 16×16 SHARED-MEMORY TILING —
  the canonical optimized GPU kernel: each 16×16 tile of C is computed by a
  workgroup that streams matching tiles of A and B through workgroup memory,
  cutting global-memory traffic by ~16×. This is where the GPU win lives."
  "
const TILE: u32 = 16u;
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec3<u32>;   // (m, k, n)
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id)        wid: vec3<u32>) {
  let m = dims.x; let k = dims.y; let n = dims.z;
  let row = wid.y * TILE + lid.y;
  let col = wid.x * TILE + lid.x;
  var acc: f32 = 0.0;
  let ntiles = (k + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
    let aCol = t * TILE + lid.x;
    let bRow = t * TILE + lid.y;
    As[lid.y][lid.x] = select(0.0, A[row * k + aCol], row < m && aCol < k);
    Bs[lid.y][lid.x] = select(0.0, B[bRow * n + col], bRow < k && col < n);
    workgroupBarrier();
    for (var i: u32 = 0u; i < TILE; i = i + 1u) {
      acc = acc + As[lid.y][i] * Bs[i][lid.x];
    }
    workgroupBarrier();
  }
  if (row < m && col < n) { C[row * n + col] = acc; }
}")

(def reduce-sum-wgsl
  "Tree reduction: each 256-wide workgroup sums its slice into `partials[wid]`;
  the host (or a second pass) sums the partials. The pattern `dot`/`nrm2`/`sum`
  share."
  "
@group(0) @binding(0) var<storage, read>       x:        array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
var<workgroup> sdata: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
  sdata[lid.x] = select(0.0, x[gid.x], gid.x < arrayLength(&x));
  workgroupBarrier();
  var s: u32 = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { partials[wid.x] = sdata[0]; }
}")

(def shaders
  "All compute kernels by op keyword — the menu a WgslBackend compiles on init."
  {:axpy   axpy-wgsl
   :spmv   spmv-csr-wgsl
   :gemm   gemm-tiled-wgsl
   :reduce reduce-sum-wgsl})

;; ---------------------------------------------------------------------------
;; IBackend ⇄ shader mapping (what a host-side WgslBackend wires up)
;; ---------------------------------------------------------------------------

(def dispatch-plan
  "For each IBackend op: the shader and how to size the grid. A WgslBackend turns
  this into (-compile) + (-dispatch) calls; ops absent here fall back to a host
  read-compute-write so the backend stays COMPLETE."
  {:-axpy  {:shader :axpy   :workgroups (fn [n] [(quot (+ n 63) 64) 1 1])}
   :-spmv  {:shader :spmv   :workgroups (fn [rows] [(quot (+ rows 63) 64) 1 1])}
   :-gemm  {:shader :gemm   :workgroups (fn [m n] [(quot (+ n 15) 16) (quot (+ m 15) 16) 1])}
   :-dot   {:shader :reduce :workgroups (fn [n] [(quot (+ n 255) 256) 1 1]) :then :host-sum}})
