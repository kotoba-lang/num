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
  (:require [clojure.string :as str]))

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

(defprotocol IGpuDeviceLifecycle
  "Optional explicit lifetime hook for device buffers."
  (-destroy-buffer [dev buffer]))

(defprotocol IGpuDeviceDType
  "Optional physical typed-buffer operations. `n` is an element count."
  (-create-buffer-dtype [dev n usage dtype])
  (-write-buffer-dtype [dev buf xs dtype])
  (-read-buffer-dtype [dev buf n dtype]))

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

(def scal-wgsl
  "Level-1 SCAL: x ← αx, in place."
  "
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             alpha: f32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&x)) { return; }
  x[i] = alpha * x[i];
}")

(def ewise-wgsl
  "Elementwise z = op(x,y); op ∈ {0:add 1:sub 2:mul 3:div} via a uniform."
  "
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       y: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform>             op: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&z)) { return; }
  let a = x[i]; let b = y[i]; var r: f32 = 0.0;
  switch op { case 0u { r = a + b; } case 1u { r = a - b; }
              case 2u { r = a * b; } case 3u { r = a / b; } default { r = 0.0; } }
  z[i] = r;
}")

(def ewise1-wgsl
  "UNARY elementwise activation/derivative selected by a uniform. Same
  shape as ewise-wgsl, one input instead of two — the primitive softmax/attention
  need that the level-1/level-2/level-3 BLAS set doesn't provide."
  "
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<uniform>             op: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&z)) { return; }
  let a = x[i]; var r: f32 = 0.0;
  switch op { case 0u { r = exp(a); } case 1u { r = max(a, 0.0); }
              case 2u { r = -a; }
              case 3u { r = a / (1.0 + exp(-a)); }
              case 4u { r = 1.0 / (1.0 + exp(-a)); }
              case 5u { r = tanh(a); }
              case 6u { r = a * (1.0 - a); }
              case 7u { r = 1.0 - a * a; }
              default { r = 0.0; } }
  z[i] = r;
}")

(def reduce-wgsl
  "Tree reduction with op ∈ {0:sum 1:max 2:min} via a uniform; each 256-wide
  workgroup writes its partial to `partials[wid]`, the host combines partials with
  the same op. `dot` = reduce(sum, ewise(mul,x,y)); `nrm2` = √(dot(x,x))."
  "
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform>             op: u32;
var<workgroup> s: array<f32, 256>;
fn cmb(a: f32, b: f32, op: u32) -> f32 {
  if (op == 1u) { return max(a, b); } if (op == 2u) { return min(a, b); } return a + b;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id)  l: vec3<u32>,
        @builtin(workgroup_id)         w: vec3<u32>) {
  let ident = select(select(3.4e38, -3.4e38, op == 1u), 0.0, op == 0u);
  s[l.x] = select(ident, x[g.x], g.x < arrayLength(&x));
  workgroupBarrier();
  var d: u32 = 128u;
  loop { if (d == 0u) { break; }
         if (l.x < d) { s[l.x] = cmb(s[l.x], s[l.x + d], op); }
         workgroupBarrier(); d = d / 2u; }
  if (l.x == 0u) { partials[w.x] = s[0]; }
}")

(def gemv-wgsl
  "Dense GEMV: y = A·x, A is m×n row-major. One thread per row."
  "
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform>             d: vec2<u32>;   // (m, n)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let m = d.x; let n = d.y;
  if (i >= m) { return; }
  var s: f32 = 0.0;
  for (var j: u32 = 0u; j < n; j = j + 1u) { s = s + A[i * n + j] * x[j]; }
  y[i] = s;
}")

(def q8-0-gemv-wgsl
  "GGML Q8_0 GEMV without an intermediate dequantized matrix. Quants are
  packed four signed bytes per u32; scales are one f32 per 32-value block."
  "
@group(0) @binding(0) var<storage, read>       q: array<u32>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: vec3<u32>; // rows, cols, blocks/row

fn signed_byte(byte_index: u32) -> f32 {
  let word = q[byte_index / 4u];
  let raw = (word >> ((byte_index % 4u) * 8u)) & 255u;
  return f32(select(i32(raw), i32(raw) - 256, raw >= 128u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.x) { return; }
  var sum: f32 = 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let scale = scales[row * d.z + block];
    let qbase = (row * d.z + block) * 32u;
    let xbase = block * 32u;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
      sum = sum + scale * signed_byte(qbase + i) * x[xbase + i];
    }
  }
  y[row] = sum;
}")

(def q4-0-gemv-wgsl
  "GGML Q4_0 GEMV without dense dequantization. Each 32-value block stores
  sixteen bytes; low nibbles are columns 0..15 and high nibbles 16..31."
  "
@group(0) @binding(0) var<storage, read>       q: array<u32>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: vec3<u32>; // rows, cols, blocks/row

fn packed_byte(byte_index: u32) -> u32 {
  let word = q[byte_index / 4u];
  return (word >> ((byte_index % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.x) { return; }
  var sum: f32 = 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let scale = scales[row * d.z + block];
    let qbase = (row * d.z + block) * 16u;
    let xbase = block * 32u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let packed = packed_byte(qbase + i);
      sum = sum + scale * (f32(packed & 15u) - 8.0) * x[xbase + i];
      sum = sum + scale * (f32(packed >> 4u) - 8.0) * x[xbase + i + 16u];
    }
  }
  y[row] = sum;
}")

(def q4-k-gemv-wgsl
  "GGML Q4_K GEMV directly over its 256-value, 144-byte superblocks."
  "
@group(0) @binding(0) var<storage, read>       raw: array<u32>;
@group(0) @binding(1) var<storage, read>       dummy: array<f32>;
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: vec3<u32>; // rows, cols, blocks/row

fn byte_at(index: u32) -> u32 {
  return (raw[index / 4u] >> ((index % 4u) * 8u)) & 255u;
}
fn group_scale(base: u32, group: u32) -> u32 {
  if (group < 4u) { return byte_at(base + 4u + group) & 63u; }
  return (byte_at(base + 8u + group) & 15u) |
         (((byte_at(base + group) >> 6u) & 3u) << 4u);
}
fn group_min(base: u32, group: u32) -> u32 {
  if (group < 4u) { return byte_at(base + 8u + group) & 63u; }
  return ((byte_at(base + 8u + group) >> 4u) & 15u) |
         (((byte_at(base + 4u + group) >> 6u) & 3u) << 4u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.x) { return; }
  var sum: f32 = dummy[0] * 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let base = (row * d.z + block) * 144u;
    let dm = unpack2x16float(raw[base / 4u]);
    for (var group: u32 = 0u; group < 8u; group = group + 1u) {
      let scale = f32(group_scale(base, group));
      let minimum = f32(group_min(base, group));
      let qbase = base + 16u + (group / 2u) * 32u;
      let xbase = block * 256u + group * 32u;
      for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let packed = byte_at(qbase + i);
        let q = select(packed & 15u, packed >> 4u, group % 2u == 1u);
        sum = sum + (dm.x * scale * f32(q) - dm.y * minimum) * x[xbase + i];
      }
    }
  }
  y[row] = sum;
}")

(def q5-0-gemv-wgsl
  "GGML Q5_0 GEMV over 32-value, 22-byte blocks."
  "
@group(0) @binding(0) var<storage, read>       raw: array<u32>;
@group(0) @binding(1) var<storage, read>       dummy: array<f32>;
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: vec3<u32>;
fn byte_at(index: u32) -> u32 {
  return (raw[index / 4u] >> ((index % 4u) * 8u)) & 255u;
}
fn high_bit(base: u32, index: u32) -> u32 {
  return ((byte_at(base + 2u + index / 8u) >> (index % 8u)) & 1u) << 4u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.x) { return; }
  var sum: f32 = dummy[0] * 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let base = (row * d.z + block) * 22u;
    let pair = unpack2x16float(raw[base / 4u]);
    let scale = select(pair.x, pair.y, base % 4u == 2u);
    let xbase = block * 32u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let packed = byte_at(base + 6u + i);
      let lo = (packed & 15u) | high_bit(base, i);
      let hi = (packed >> 4u) | high_bit(base, i + 16u);
      sum = sum + scale * (f32(lo) - 16.0) * x[xbase + i];
      sum = sum + scale * (f32(hi) - 16.0) * x[xbase + i + 16u];
    }
  }
  y[row] = sum;
}")

(def q5-1-gemv-wgsl
  "GGML Q5_1 GEMV over 32-value, 24-byte blocks."
  "
@group(0) @binding(0) var<storage, read>       raw: array<u32>;
@group(0) @binding(1) var<storage, read>       dummy: array<f32>;
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: vec3<u32>;
fn byte_at(index: u32) -> u32 {
  return (raw[index / 4u] >> ((index % 4u) * 8u)) & 255u;
}
fn high_bit(base: u32, index: u32) -> u32 {
  return ((byte_at(base + 4u + index / 8u) >> (index % 8u)) & 1u) << 4u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.x) { return; }
  var sum: f32 = dummy[0] * 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let base = (row * d.z + block) * 24u;
    let dm = unpack2x16float(raw[base / 4u]);
    let xbase = block * 32u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let packed = byte_at(base + 8u + i);
      let lo = (packed & 15u) | high_bit(base, i);
      let hi = (packed >> 4u) | high_bit(base, i + 16u);
      sum = sum + (dm.x * f32(lo) + dm.y) * x[xbase + i];
      sum = sum + (dm.x * f32(hi) + dm.y) * x[xbase + i + 16u];
    }
  }
  y[row] = sum;
}")

(def q5-k-gemv-wgsl
  "GGML Q5_K GEMV directly over 176-byte superblocks."
  "
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> dummy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: vec3<u32>;
fn byte_at(i: u32) -> u32 { return (raw[i / 4u] >> ((i % 4u) * 8u)) & 255u; }
fn gs(b: u32, g: u32) -> u32 {
  if (g < 4u) { return byte_at(b + 4u + g) & 63u; }
  return (byte_at(b + 8u + g) & 15u) | (((byte_at(b + g) >> 6u) & 3u) << 4u);
}
fn gm(b: u32, g: u32) -> u32 {
  if (g < 4u) { return byte_at(b + 8u + g) & 63u; }
  return ((byte_at(b + 8u + g) >> 4u) & 15u) |
         (((byte_at(b + 4u + g) >> 6u) & 3u) << 4u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; if (row >= d.x) { return; }
  var sum: f32 = dummy[0] * 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let b = (row * d.z + block) * 176u;
    let dm = unpack2x16float(raw[b / 4u]);
    for (var g: u32 = 0u; g < 8u; g = g + 1u) {
      for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let packed = byte_at(b + 48u + (g / 2u) * 32u + i);
        let low = select(packed & 15u, packed >> 4u, g % 2u == 1u);
        let high = ((byte_at(b + 16u + i) >> g) & 1u) << 4u;
        let q = low | high;
        sum = sum + (dm.x * f32(gs(b, g)) * f32(q) - dm.y * f32(gm(b, g))) *
                    x[block * 256u + g * 32u + i];
      }
    }
  }
  y[row] = sum;
}")

(def q6-k-gemv-wgsl
  "GGML Q6_K GEMV directly over signed 6-bit, 210-byte superblocks."
  "
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> dummy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: vec3<u32>;
fn byte_at(i: u32) -> u32 { return (raw[i / 4u] >> ((i % 4u) * 8u)) & 255u; }
fn signed_byte(i: u32) -> i32 {
  let v = byte_at(i); return select(i32(v), i32(v) - 256, v >= 128u);
}
fn half_at(i: u32) -> f32 {
  let p = unpack2x16float(raw[i / 4u]); return select(p.x, p.y, i % 4u == 2u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; if (row >= d.x) { return; }
  var sum: f32 = dummy[0] * 0.0;
  for (var block: u32 = 0u; block < d.z; block = block + 1u) {
    let b = (row * d.z + block) * 210u; let scale = half_at(b + 208u);
    for (var hb: u32 = 0u; hb < 2u; hb = hb + 1u) {
      for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let la = byte_at(b + hb * 64u + i);
        let lb = byte_at(b + hb * 64u + 32u + i);
        let hi = byte_at(b + 128u + hb * 32u + i);
        let qs = array<i32, 4>(i32((la & 15u) | ((hi & 3u) << 4u)) - 32,
          i32((lb & 15u) | (((hi >> 2u) & 3u) << 4u)) - 32,
          i32((la >> 4u) | (((hi >> 4u) & 3u) << 4u)) - 32,
          i32((lb >> 4u) | (((hi >> 6u) & 3u) << 4u)) - 32);
        for (var seg: u32 = 0u; seg < 4u; seg = seg + 1u) {
          let si = hb * 8u + seg * 2u + i / 16u;
          let weight = scale * f32(signed_byte(b + 192u + si)) * f32(qs[seg]);
          sum = sum + weight * x[block * 256u + hb * 128u + seg * 32u + i];
        }
      }
    }
  }
  y[row] = sum;
}")

(def kv-cache-write-wgsl
  "Write the current token's RoPE-transformed K/V vectors into a persistent,
  layer-major KV cache."
  "
struct Params { heads: u32, kv_heads: u32, head_dim: u32, position: u32,
                context: u32, layer: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> k: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys: array<f32>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let width = p.kv_heads * p.head_dim;
  if (i >= width) { return; }
  let base = (p.layer * p.context + p.position) * width;
  keys[base + i] = k[i]; values[base + i] = v[i];
}")

(def causal-gqa-attention-wgsl
  "Causal grouped-query attention over a persistent KV cache. Each invocation
  computes one output component using stable online softmax passes."
  "
struct Params { heads: u32, kv_heads: u32, head_dim: u32, position: u32,
                context: u32, layer: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
fn score(head: u32, kv_head: u32, past: u32) -> f32 {
  let width = p.kv_heads * p.head_dim;
  let kb = (p.layer * p.context + past) * width + kv_head * p.head_dim;
  let qb = head * p.head_dim; var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < p.head_dim; i = i + 1u) {
    sum = sum + q[qb + i] * keys[kb + i];
  }
  return sum * inverseSqrt(f32(p.head_dim));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x; if (index >= p.heads * p.head_dim) { return; }
  let head = index / p.head_dim; let component = index % p.head_dim;
  let grouped = p.heads / p.kv_heads; let kv_head = head / grouped;
  var maximum: f32 = -3.402823e38;
  for (var past: u32 = 0u; past <= p.position; past = past + 1u) {
    maximum = max(maximum, score(head, kv_head, past));
  }
  var denominator: f32 = 0.0; var result: f32 = 0.0;
  let width = p.kv_heads * p.head_dim;
  for (var past: u32 = 0u; past <= p.position; past = past + 1u) {
    let weight = exp(score(head, kv_head, past) - maximum);
    let vb = (p.layer * p.context + past) * width + kv_head * p.head_dim;
    denominator = denominator + weight;
    result = result + weight * values[vb + component];
  }
  output[index] = result / denominator;
}")

(def transpose-2d-wgsl
  "Out-of-place row-major matrix transpose used by GPU backpropagation."
  "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x; let row = gid.y;
  if (row < dims.x && col < dims.y) {
    output[col * dims.x + row] = input[row * dims.y + col];
  }
}")

(def add-last-axis-bias-wgsl
  "Broadcast a rank-1 bias over contiguous rows."
  "
struct Params { total: u32, width: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.total) { return; }
  output[i] = input[i] + bias[i % p.width];
}")

(def multi-head-attention-wgsl
  "Fused rank-2 Q/K/V multi-head attention with stable online softmax passes."
  "
struct Params { batch: u32, seq_q: u32, seq_k: u32, model: u32, heads: u32,
                head_dim: u32, total: u32, causal: u32, has_mask: u32,
                pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> key_padding_mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
fn allowed(batch: u32, row: u32, token: u32) -> bool {
  return !((p.causal != 0u && token > row) ||
           (p.has_mask != 0u && key_padding_mask[batch * p.seq_k + token] != 0.0));
}
fn score(batch: u32, row: u32, head: u32, token: u32) -> f32 {
  let qbase = (batch * p.seq_q + row) * p.model + head * p.head_dim;
  let kbase = (batch * p.seq_k + token) * p.model + head * p.head_dim;
  var sum: f32 = 0.0;
  for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
    sum = sum + query[qbase + d] * key[kbase + d];
  }
  return sum * inverseSqrt(f32(p.head_dim));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.total) { return; }
  let flat_row = i / p.model; let batch = flat_row / p.seq_q;
  let row = flat_row % p.seq_q; let component = i % p.model;
  let head = component / p.head_dim;
  var maximum: f32 = -3.402823e38;
  for (var token: u32 = 0u; token < p.seq_k; token = token + 1u) {
    if (allowed(batch, row, token)) {
      maximum = max(maximum, score(batch, row, head, token));
    }
  }
  var denominator: f32 = 0.0; var result: f32 = 0.0;
  for (var token: u32 = 0u; token < p.seq_k; token = token + 1u) {
    if (allowed(batch, row, token)) {
      let weight = exp(score(batch, row, head, token) - maximum);
      denominator = denominator + weight;
      result = result + weight * value[(batch * p.seq_k + token) * p.model + component];
    }
  }
  output[i] = select(0.0, result / denominator, denominator > 0.0);
}")

(def multi-head-attention-backward-wgsl
  "Recompute stable softmax and produce Q/K/V gradients without host tensors."
  "
struct Params { batch: u32, seq_q: u32, seq_k: u32, model: u32, heads: u32,
                head_dim: u32, total_q: u32, total_k: u32, total: u32,
                causal: u32, has_mask: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> key_padding_mask: array<f32>;
@group(0) @binding(4) var<storage, read> grad_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_query: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_key: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_value: array<f32>;
@group(0) @binding(8) var<uniform> p: Params;
fn allowed(batch: u32, row: u32, token: u32) -> bool {
  return !((p.causal != 0u && token > row) ||
           (p.has_mask != 0u && key_padding_mask[batch * p.seq_k + token] != 0.0));
}
fn score(batch: u32, row: u32, head: u32, token: u32) -> f32 {
  let qb = (batch * p.seq_q + row) * p.model + head * p.head_dim;
  let kb = (batch * p.seq_k + token) * p.model + head * p.head_dim;
  var sum: f32 = 0.0;
  for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
    sum = sum + query[qb + d] * key[kb + d];
  }
  return sum * inverseSqrt(f32(p.head_dim));
}
fn softmax_max(batch: u32, row: u32, head: u32) -> f32 {
  var maximum: f32 = -3.402823e38;
  for (var k: u32 = 0u; k < p.seq_k; k = k + 1u) {
    if (allowed(batch, row, k)) {
      maximum = max(maximum, score(batch, row, head, k));
    }
  }
  return maximum;
}
fn softmax_denominator(batch: u32, row: u32, head: u32, maximum: f32) -> f32 {
  var denominator: f32 = 0.0;
  for (var k: u32 = 0u; k < p.seq_k; k = k + 1u) {
    if (allowed(batch, row, k)) {
      denominator = denominator + exp(score(batch, row, head, k) - maximum);
    }
  }
  return denominator;
}
fn probability(batch: u32, row: u32, head: u32, token: u32,
               maximum: f32, denominator: f32) -> f32 {
  if (!allowed(batch, row, token) || denominator == 0.0) { return 0.0; }
  return exp(score(batch, row, head, token) - maximum) / denominator;
}
fn grad_probability(batch: u32, row: u32, head: u32, token: u32) -> f32 {
  let ob = (batch * p.seq_q + row) * p.model + head * p.head_dim;
  let vb = (batch * p.seq_k + token) * p.model + head * p.head_dim;
  var sum: f32 = 0.0;
  for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
    sum = sum + grad_output[ob + d] * value[vb + d];
  }
  return sum;
}
fn grad_expectation(batch: u32, row: u32, head: u32,
                    maximum: f32, denominator: f32) -> f32 {
  var expected: f32 = 0.0;
  for (var k: u32 = 0u; k < p.seq_k; k = k + 1u) {
    expected = expected + probability(batch, row, head, k, maximum, denominator) *
                          grad_probability(batch, row, head, k);
  }
  return expected;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let scale = inverseSqrt(f32(p.head_dim));
  if (i < p.total_q) {
    let flat_row = i / p.model; let batch = flat_row / p.seq_q;
    let row = flat_row % p.seq_q; let component = i % p.model;
    let head = component / p.head_dim;
    let maximum = softmax_max(batch, row, head);
    let denominator = softmax_denominator(batch, row, head, maximum);
    let expected = grad_expectation(batch, row, head, maximum, denominator);
    var dq: f32 = 0.0;
    for (var token: u32 = 0u; token < p.seq_k; token = token + 1u) {
      let prob = probability(batch, row, head, token, maximum, denominator);
      let ds = prob * (grad_probability(batch, row, head, token) - expected);
      dq = dq + ds * key[(batch * p.seq_k + token) * p.model + component] * scale;
    }
    grad_query[i] = dq;
  }
  if (i < p.total_k) {
    let flat_token = i / p.model; let batch = flat_token / p.seq_k;
    let token = flat_token % p.seq_k; let component = i % p.model;
    let head = component / p.head_dim; var dk: f32 = 0.0; var dv: f32 = 0.0;
    for (var row: u32 = 0u; row < p.seq_q; row = row + 1u) {
      let maximum = softmax_max(batch, row, head);
      let denominator = softmax_denominator(batch, row, head, maximum);
      let prob = probability(batch, row, head, token, maximum, denominator);
      let expected = grad_expectation(batch, row, head, maximum, denominator);
      let ds = prob * (grad_probability(batch, row, head, token) - expected);
      dk = dk + ds * query[(batch * p.seq_q + row) * p.model + component] * scale;
      dv = dv + prob * grad_output[(batch * p.seq_q + row) * p.model + component];
    }
    grad_key[i] = dk; grad_value[i] = dv;
  }
}")

(def add-bias-rows-wgsl
  "Add one shared bias vector to every matrix row."
  "
@group(0) @binding(0) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= dims.x * dims.y) { return; }
  matrix[i] = matrix[i] + bias[i % dims.y];
}")

(def mse-loss-wgsl
  "Mean squared error reduced into one device-resident scalar."
  "
@group(0) @binding(0) var<storage, read> prediction: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
@group(0) @binding(3) var<uniform> count: u32;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  var sum: f32 = 0.0;
  for (var i: u32 = lid; i < count; i = i + 64u) {
    let difference = prediction[i] - expected[i];
    sum = sum + difference * difference;
  }
  partial[lid] = sum;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) { partial[lid] = partial[lid] + partial[lid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid == 0u) { loss[0] = partial[0] / f32(count); }
}")

(def mse-gradient-wgsl
  "MSE vector-Jacobian product, including the upstream scalar seed."
  "
@group(0) @binding(0) var<storage, read> prediction: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read> upstream: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(4) var<uniform> count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= count) { return; }
  gradient[i] = upstream[0] * 2.0 * (prediction[i] - expected[i]) / f32(count);
}")

(def relu-backward-wgsl
  "ReLU vector-Jacobian product using saved pre-activation values."
  "
@group(0) @binding(0) var<storage, read> upstream: array<f32>;
@group(0) @binding(1) var<storage, read> preactivation: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(3) var<uniform> count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= count) { return; }
  gradient[i] = select(0.0, upstream[i], preactivation[i] > 0.0);
}")

(def bias-gradient-wgsl
  "Sum a matrix gradient over rows into its shared bias gradient."
  "
@group(0) @binding(0) var<storage, read> gradient: array<f32>;
@group(0) @binding(1) var<storage, read_write> bias_gradient: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x; if (col >= dims.y) { return; }
  var sum: f32 = 0.0;
  for (var row: u32 = 0u; row < dims.x; row = row + 1u) {
    sum = sum + gradient[row * dims.y + col];
  }
  bias_gradient[col] = sum;
}")

(def sgd-update-wgsl
  "In-place SGD parameter update."
  "
@group(0) @binding(0) var<storage, read_write> parameter: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
struct Params { learning_rate: f32, count: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.count) { return; }
  parameter[i] = parameter[i] - p.learning_rate * gradient[i];
}")

(def sgd-step-wgsl
  "Out-of-place parameter - learning_rate * gradient update."
  "
@group(0) @binding(0) var<storage, read> parameter: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> learning_rate: f32;
@group(0) @binding(4) var<uniform> count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= count) { return; }
  output[i] = parameter[i] - learning_rate * gradient[i];
}")

(def adamw-step-wgsl
  "Fused out-of-place AdamW parameter and optimizer-slot update."
  "
@group(0) @binding(0) var<storage, read> parameter: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read> moment: array<f32>;
@group(0) @binding(3) var<storage, read> variance: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_parameter: array<f32>;
@group(0) @binding(5) var<storage, read_write> next_moment: array<f32>;
@group(0) @binding(6) var<storage, read_write> next_variance: array<f32>;
struct Hyper {
  learning_rate: f32, beta1: f32, beta2: f32, eps: f32,
  weight_decay: f32, correction1: f32, correction2: f32, pad0: f32,
}
@group(0) @binding(7) var<uniform> h: Hyper;
@group(0) @binding(8) var<uniform> count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= count) { return; }
  let m = h.beta1 * moment[i] + (1.0 - h.beta1) * gradient[i];
  let v = h.beta2 * variance[i] + (1.0 - h.beta2) * gradient[i] * gradient[i];
  let adaptive = (m / h.correction1) /
                 (sqrt(v / h.correction2) + h.eps);
  next_moment[i] = m;
  next_variance[i] = v;
  next_parameter[i] = parameter[i] - h.learning_rate *
                      (adaptive + h.weight_decay * parameter[i]);
}")

(def unscale-gradient-wgsl
  "Fused loss-scale removal and atomic non-finite detection."
  "
@group(0) @binding(0) var<storage, read> gradient: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> found_inf: atomic<u32>;
@group(0) @binding(3) var<uniform> inverse_scale: f32;
@group(0) @binding(4) var<uniform> count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= count) { return; }
  let value = gradient[i];
  if (value != value || abs(value) > 3.402823e+38) {
    atomicOr(&found_inf, 1u);
  }
  output[i] = value * inverse_scale;
}")

(def conv2d-nchw-wgsl
  "Direct NCHW convolution/cross-correlation. One invocation computes one
  output element; supports bias, groups/depthwise, stride, padding, dilation."
  "
struct Params {
  n: u32, cin: u32, h: u32, w: u32,
  cout: u32, cin_group: u32, kh: u32, kw: u32,
  oh: u32, ow: u32, sh: u32, sw: u32,
  ph: u32, pw: u32, dh: u32, dw: u32,
  groups: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = p.n * p.cout * p.oh * p.ow;
  if (index >= total) { return; }
  let oj = index % p.ow;
  let oi = (index / p.ow) % p.oh;
  let oc = (index / (p.ow * p.oh)) % p.cout;
  let batch = index / (p.ow * p.oh * p.cout);
  let outputs_per_group = p.cout / p.groups;
  let group = oc / outputs_per_group;
  let input_channel_base = group * p.cin_group;
  var sum = bias[oc];
  for (var icg: u32 = 0u; icg < p.cin_group; icg = icg + 1u) {
    let ic = input_channel_base + icg;
    for (var ki: u32 = 0u; ki < p.kh; ki = ki + 1u) {
      let ih = i32(oi * p.sh + ki * p.dh) - i32(p.ph);
      if (ih < 0 || ih >= i32(p.h)) { continue; }
      for (var kj: u32 = 0u; kj < p.kw; kj = kj + 1u) {
        let iw = i32(oj * p.sw + kj * p.dw) - i32(p.pw);
        if (iw < 0 || iw >= i32(p.w)) { continue; }
        let x_index = ((batch * p.cin + ic) * p.h + u32(ih)) * p.w + u32(iw);
        let w_index = ((oc * p.cin_group + icg) * p.kh + ki) * p.kw + kj;
        sum = sum + input[x_index] * weight[w_index];
      }
    }
  }
  output[index] = sum;
}")

(def conv2d-nchw-oc4-wgsl
  "Groups=1 fast path computing four output channels per invocation so each
  input element and coordinate calculation is reused across four FMAs."
  "
struct Params {
  n: u32, cin: u32, h: u32, w: u32,
  cout: u32, cin_group: u32, kh: u32, kw: u32,
  oh: u32, ow: u32, sh: u32, sw: u32,
  ph: u32, pw: u32, dh: u32, dw: u32,
  groups: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let spatial = p.oh * p.ow;
  let cout4 = p.cout / 4u;
  let total = p.n * cout4 * spatial;
  if (index >= total) { return; }
  let position = index % spatial;
  let oj = position % p.ow;
  let oi = position / p.ow;
  let oc_block = (index / spatial) % cout4;
  let batch = index / (spatial * cout4);
  let oc = oc_block * 4u;
  var sum = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);
  let kernel_size = p.cin * p.kh * p.kw;
  for (var ic: u32 = 0u; ic < p.cin; ic = ic + 1u) {
    for (var ki: u32 = 0u; ki < p.kh; ki = ki + 1u) {
      let ih = i32(oi * p.sh + ki * p.dh) - i32(p.ph);
      if (ih < 0 || ih >= i32(p.h)) { continue; }
      for (var kj: u32 = 0u; kj < p.kw; kj = kj + 1u) {
        let iw = i32(oj * p.sw + kj * p.dw) - i32(p.pw);
        if (iw < 0 || iw >= i32(p.w)) { continue; }
        let kernel_index = (ic * p.kh + ki) * p.kw + kj;
        let x_index = ((batch * p.cin + ic) * p.h + u32(ih)) * p.w + u32(iw);
        let x = input[x_index];
        let weights = vec4<f32>(weight[oc * kernel_size + kernel_index],
                                weight[(oc + 1u) * kernel_size + kernel_index],
                                weight[(oc + 2u) * kernel_size + kernel_index],
                                weight[(oc + 3u) * kernel_size + kernel_index]);
        sum = sum + vec4<f32>(x) * weights;
      }
    }
  }
  output[(batch * p.cout + oc) * spatial + position] = sum.x;
  output[(batch * p.cout + oc + 1u) * spatial + position] = sum.y;
  output[(batch * p.cout + oc + 2u) * spatial + position] = sum.z;
  output[(batch * p.cout + oc + 3u) * spatial + position] = sum.w;
}")

(def group-norm-nchw-wgsl
  "Parallel GroupNorm: one 256-thread workgroup reduces and normalizes one
  `[channels/group,H,W]` slice. Variance uses the biased PyTorch definition."
  "
struct Dims {
  n: u32, c: u32, h: u32, w: u32,
  groups: u32, channels_group: u32, group_size: u32, spatial: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<uniform> epsilon: f32;
var<workgroup> sums: array<f32, 256>;
var<workgroup> squares: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(workgroup_id) wid3: vec3<u32>) {
  let lid = lid3.x;
  let slice = wid3.x;
  if (slice >= d.n * d.groups) { return; }
  let batch = slice / d.groups;
  let group = slice % d.groups;
  let base = batch * d.c * d.h * d.w + group * d.group_size;
  var sum = 0.0;
  var sum_square = 0.0;
  for (var i = lid; i < d.group_size; i = i + 256u) {
    let value = input[base + i];
    sum = sum + value;
    sum_square = sum_square + value * value;
  }
  sums[lid] = sum;
  squares[lid] = sum_square;
  workgroupBarrier();
  var offset = 128u;
  loop {
    if (offset == 0u) { break; }
    if (lid < offset) {
      sums[lid] = sums[lid] + sums[lid + offset];
      squares[lid] = squares[lid] + squares[lid + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }
  let mean = sums[0] / f32(d.group_size);
  let variance = max(squares[0] / f32(d.group_size) - mean * mean, 0.0);
  let inv_std = inverseSqrt(variance + epsilon);
  for (var i = lid; i < d.group_size; i = i + 256u) {
    let channel = group * d.channels_group + i / d.spatial;
    output[base + i] = (input[base + i] - mean) * inv_std * weight[channel] + bias[channel];
  }
}")

(def group-norm-silu-nchw-wgsl
  (str/replace
   group-norm-nchw-wgsl
   "output[base + i] = (input[base + i] - mean) * inv_std * weight[channel] + bias[channel];"
   "let normalized = (input[base + i] - mean) * inv_std * weight[channel] + bias[channel];\n    output[base + i] = normalized / (1.0 + exp(-normalized));"))

(def upsample-nearest2d-wgsl
  "Nearest-neighbor NCHW upsampling, one invocation per output element."
  "
struct Dims {
  n: u32, c: u32, h: u32, w: u32,
  oh: u32, ow: u32, scale_h: u32, scale_w: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = d.n * d.c * d.oh * d.ow;
  if (index >= total) { return; }
  let oj = index % d.ow;
  let oi = (index / d.ow) % d.oh;
  let channel = (index / (d.ow * d.oh)) % d.c;
  let batch = index / (d.ow * d.oh * d.c);
  let input_index = ((batch * d.c + channel) * d.h + oi / d.scale_h) * d.w
                    + oj / d.scale_w;
  output[index] = input[input_index];
}")

(def cat-copy-wgsl
  "Copy one contiguous tensor into its slice of a concatenated output. Repeated
  queue-ordered dispatches fill one output buffer without host readback."
  "
struct Dims {
  total: u32, block: u32, output_block: u32, axis_offset: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= d.total) { return; }
  let outer = index / d.block;
  let within = index % d.block;
  output[outer * d.output_block + d.axis_offset + within] = input[index];
}")

(def slice-axis-wgsl
  "Copy a contiguous range from each flattened outer block."
  "
struct Dims {
  total: u32, input_block: u32, output_block: u32, input_offset: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= d.total) { return; }
  let outer = index / d.output_block;
  let within = index % d.output_block;
  output[index] = input[outer * d.input_block + d.input_offset + within];
}")

(def pad-right-bottom-nchw-wgsl
  "Append a zero-valued right column and bottom row to every NCHW plane."
  "
struct Dims { total: u32, h: u32, w: u32, output_w: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= d.total) { return; }
  let x = index % d.output_w;
  let y = (index / d.output_w) % (d.h + 1u);
  if (x == d.w || y == d.h) {
    output[index] = 0.0;
  } else {
    let plane = index / (d.output_w * (d.h + 1u));
    output[index] = input[(plane * d.h + y) * d.w + x];
  }
}")

(def ewise-f16-wgsl
  "Packed physical f16 elementwise arithmetic with f32 evaluation."
  "
struct Params { op: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> y: array<u32>;
@group(0) @binding(2) var<storage, read_write> z: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word = gid.x;
  let base = word * 2u;
  if (base >= p.n) { return; }
  let xv = unpack2x16float(x[word]);
  let yv = unpack2x16float(y[word]);
  var out: vec2<f32>;
  if (p.op == 0u) { out = xv + yv; }
  else if (p.op == 1u) { out = xv - yv; }
  else if (p.op == 2u) { out = xv * yv; }
  else { out = xv / yv; }
  if (base + 1u >= p.n) { out.y = 0.0; }
  z[word] = pack2x16float(out);
}")

(def ewise1-f16-wgsl
  "Packed physical f16 unary ops, with transcendental evaluation in f32."
  "
struct Params { op: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> z: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;
fn apply(v: f32) -> f32 {
  if (p.op == 0u) { return exp(v); }
  if (p.op == 1u) { return max(v, 0.0); }
  if (p.op == 2u) { return -v; }
  if (p.op == 3u) { return v / (1.0 + exp(-v)); }
  if (p.op == 4u) { return 1.0 / (1.0 + exp(-v)); }
  if (p.op == 5u) { return tanh(v); }
  if (p.op == 6u) { return v * (1.0 - v); }
  return 1.0 - v * v;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word = gid.x;
  let base = word * 2u;
  if (base >= p.n) { return; }
  let v = unpack2x16float(x[word]);
  let second = select(apply(v.y), 0.0, base + 1u >= p.n);
  z[word] = pack2x16float(vec2<f32>(apply(v.x), second));
}")

(def gemm-f16-wgsl
  "Packed physical f16 GEMM with f32 accumulation and f16 output."
  "
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
@group(0) @binding(3) var<uniform> d: Dims;
fn load_a(index: u32) -> f32 {
  let pair = unpack2x16float(A[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn load_b(index: u32) -> f32 {
  let pair = unpack2x16float(B[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn dot_at(index: u32) -> f32 {
  let row = index / d.n;
  let col = index % d.n;
  var sum: f32 = 0.0;
  for (var l: u32 = 0u; l < d.k; l = l + 1u) {
    sum = sum + load_a(row * d.k + l) * load_b(l * d.n + col);
  }
  return sum;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word = gid.x;
  let base = word * 2u;
  let total = d.m * d.n;
  if (base >= total) { return; }
  var second: f32 = 0.0;
  if (base + 1u < total) { second = dot_at(base + 1u); }
  C[word] = pack2x16float(vec2<f32>(dot_at(base), second));
}")

(def conv2d-nchw-f16-wgsl
  "Packed-f16 NCHW grouped convolution with f32 accumulation."
  "
struct Params {
  n: u32, cin: u32, h: u32, w: u32,
  cout: u32, cin_group: u32, kh: u32, kw: u32,
  oh: u32, ow: u32, sh: u32, sw: u32,
  ph: u32, pw: u32, dh: u32, dw: u32,
  groups: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> p: Params;
fn load_input(index: u32) -> f32 {
  let pair = unpack2x16float(input[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn load_weight(index: u32) -> f32 {
  let pair = unpack2x16float(weight[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn load_bias(index: u32) -> f32 {
  let pair = unpack2x16float(bias[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn convolve(index: u32) -> f32 {
  let oj = index % p.ow;
  let oi = (index / p.ow) % p.oh;
  let oc = (index / (p.ow * p.oh)) % p.cout;
  let batch = index / (p.ow * p.oh * p.cout);
  let outputs_per_group = p.cout / p.groups;
  let input_channel_base = (oc / outputs_per_group) * p.cin_group;
  var sum = load_bias(oc);
  for (var icg = 0u; icg < p.cin_group; icg = icg + 1u) {
    let ic = input_channel_base + icg;
    for (var ki = 0u; ki < p.kh; ki = ki + 1u) {
      let ih = i32(oi * p.sh + ki * p.dh) - i32(p.ph);
      if (ih < 0 || ih >= i32(p.h)) { continue; }
      for (var kj = 0u; kj < p.kw; kj = kj + 1u) {
        let iw = i32(oj * p.sw + kj * p.dw) - i32(p.pw);
        if (iw < 0 || iw >= i32(p.w)) { continue; }
        let xi = ((batch * p.cin + ic) * p.h + u32(ih)) * p.w + u32(iw);
        let wi = ((oc * p.cin_group + icg) * p.kh + ki) * p.kw + kj;
        sum = sum + load_input(xi) * load_weight(wi);
      }
    }
  }
  return sum;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * 2u;
  let total = p.n * p.cout * p.oh * p.ow;
  if (base >= total) { return; }
  var second = 0.0;
  if (base + 1u < total) { second = convolve(base + 1u); }
  output[gid.x] = pack2x16float(vec2<f32>(convolve(base), second));
}")

(def group-norm-nchw-f16-wgsl
  "Packed-f16 GroupNorm reference kernel with f32 statistics."
  "
struct Dims {
  n: u32, c: u32, h: u32, w: u32,
  groups: u32, channels_group: u32, group_size: u32, spatial: u32,
}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<uniform> epsilon: f32;
fn load_input_value(index: u32) -> f32 {
  let pair = unpack2x16float(input[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn load_weight_value(index: u32) -> f32 {
  let pair = unpack2x16float(weight[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn load_bias_value(index: u32) -> f32 {
  let pair = unpack2x16float(bias[index / 2u]);
  return select(pair.x, pair.y, index % 2u == 1u);
}
fn normalize(index: u32) -> f32 {
  let batch = index / (d.c * d.h * d.w);
  let channel = (index / d.spatial) % d.c;
  let group = channel / d.channels_group;
  let base = batch * d.c * d.h * d.w + group * d.group_size;
  var sum = 0.0;
  var sum_square = 0.0;
  for (var i = 0u; i < d.group_size; i = i + 1u) {
    let v = load_input_value(base + i);
    sum = sum + v;
    sum_square = sum_square + v * v;
  }
  let mean = sum / f32(d.group_size);
  let variance = max(sum_square / f32(d.group_size) - mean * mean, 0.0);
  return (load_input_value(index) - mean) * inverseSqrt(variance + epsilon)
         * load_weight_value(channel) + load_bias_value(channel);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * 2u;
  let total = d.n * d.c * d.h * d.w;
  if (base >= total) { return; }
  var second = 0.0;
  if (base + 1u < total) { second = normalize(base + 1u); }
  output[gid.x] = pack2x16float(vec2<f32>(normalize(base), second));
}")

(def shaders
  "All compute kernels by op keyword — the menu a WgslBackend compiles on init.
  Verified on Apple M4 Metal (wgpu via WebGPU): the full IBackend contract
  (axpy/scal/dot/nrm2/ewise/reduce/gemv/gemm/spmv) reproduces the CPU reference —
  see verify/metal_contract.js."
  {:axpy   axpy-wgsl
   :scal   scal-wgsl
   :ewise  ewise-wgsl
   :ewise1 ewise1-wgsl
   :reduce reduce-wgsl
   :gemv   gemv-wgsl
   :gemm   gemm-tiled-wgsl
   :conv2d-nchw conv2d-nchw-wgsl
   :conv2d-nchw-oc4 conv2d-nchw-oc4-wgsl
   :group-norm-nchw group-norm-nchw-wgsl
   :group-norm-silu-nchw group-norm-silu-nchw-wgsl
   :upsample-nearest2d upsample-nearest2d-wgsl
   :cat-copy cat-copy-wgsl
   :slice-axis slice-axis-wgsl
   :pad-right-bottom-nchw pad-right-bottom-nchw-wgsl
   :add-last-axis-bias add-last-axis-bias-wgsl
   :transpose-2d transpose-2d-wgsl
   :bias-gradient bias-gradient-wgsl
   :mse-loss mse-loss-wgsl
   :mse-gradient mse-gradient-wgsl
   :sgd-step sgd-step-wgsl
   :adamw-step adamw-step-wgsl
   :unscale-gradient unscale-gradient-wgsl
   :multi-head-attention multi-head-attention-wgsl
   :multi-head-attention-backward multi-head-attention-backward-wgsl
   :ewise-f16 ewise-f16-wgsl
   :ewise1-f16 ewise1-f16-wgsl
   :gemm-f16 gemm-f16-wgsl
   :conv2d-nchw-f16 conv2d-nchw-f16-wgsl
   :group-norm-nchw-f16 group-norm-nchw-f16-wgsl
   :spmv   spmv-csr-wgsl})

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
