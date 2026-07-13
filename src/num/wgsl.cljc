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
  "UNARY elementwise z = op(x); op ∈ {0:exp 1:relu 2:neg 3:silu} via a uniform. Same
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
  return v / (1.0 + exp(-v));
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
   :group-norm-nchw group-norm-nchw-wgsl
   :upsample-nearest2d upsample-nearest2d-wgsl
   :cat-copy cat-copy-wgsl
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
