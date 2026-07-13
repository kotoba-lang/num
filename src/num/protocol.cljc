(ns num.protocol
  "IBackend — the ONE seam every compute backend implements, and the whole reason
  num-clj can be `.cljc`-portable yet GPU-fast.

  A `handle` is an opaque per-backend buffer reference: a `double-array` for the
  CPU reference backend, a `GPUBuffer` for the WGSL/WebGPU backend, a device
  pointer for CUDA/Metal/ROCm. The array layer (`num.array`) never touches a
  handle's internals — it only moves them through these methods. So the SAME
  high-level program (`num.core/matmul`, `spmv`, …) runs unchanged whether the
  active backend is pure Clojure on a phone, a WGSL compute pipeline in a browser
  tab, or cuBLAS on an H100 — exactly the Store-injection / cae-solver-dispatch
  pattern used across com-junkawasaki.

  Every op is EAGER and explicit (no lazy graph at this layer); dims travel
  alongside handles BLAS-style so a backend never has to infer shape. Dense
  matrices are row-major. Reductions/dot return host scalars; everything else
  returns a handle (often mutated in place, BLAS convention).")

(defprotocol IBackend
  ;; identity / lifecycle ----------------------------------------------------
  (-backend-name [b] "Keyword id, e.g. :cpu / :wgsl / :cuda.")
  (-alloc [b n] "Allocate an uninitialized buffer of `n` scalars → handle.")
  (-free [b h] "Release a handle (no-op where GC handles it).")
  (-copy-to-host [b h n] "Device→host: return a Clojure vector of `n` doubles.")
  (-copy-from-host [b xs] "Host→device: `xs` (seq of numbers) → handle.")

  ;; level-1 BLAS ------------------------------------------------------------
  (-axpy [b alpha xh yh n] "y ← αx + y (in place on yh); returns yh.")
  (-scal [b alpha xh n] "x ← αx (in place); returns xh.")
  (-dot [b xh yh n] "Σ xᵢ yᵢ → host scalar.")
  (-nrm2 [b xh n] "‖x‖₂ → host scalar.")

  ;; elementwise + reduction -------------------------------------------------
  (-ewise [b op xh yh n] "op ∈ #{:add :sub :mul :div}; returns a NEW handle z=op(x,y).")
  (-ewise1 [b op xh n] "UNARY elementwise; op ∈ #{:exp :relu :neg :silu}; returns a NEW handle z=op(x).")
  (-reduce [b op xh n] "op ∈ #{:sum :max :min}; → host scalar.")

  ;; level-2 / level-3 BLAS (dense, row-major) -------------------------------
  (-gemv [b alpha Ah m n xh beta yh] "y ← αAx + βy, A is m×n; returns yh.")
  (-gemm [b alpha Ah m k Bh n beta Ch] "C ← αAB + βC, A m×k, B k×n, C m×n; returns Ch.")

  ;; sparse (CSR) ------------------------------------------------------------
  (-spmv [b csr xh] "Sparse mat-vec y = A·x for a CSR `csr` (see num.sparse); → new handle."))

(defn backend?
  "Does `x` satisfy IBackend?"
  [x]
  (satisfies? IBackend x))

(defprotocol ITensorBackend
  "Optional device-native N-D operations. Backends that do not implement this
  protocol continue to use num.tensor's portable host oracle."
  (-conv2d-nchw [b input-h weight-h bias-h params]
    "NCHW cross-correlation. `params` contains validated dimensions/options;
    returns a newly allocated output handle.")
  (-group-norm-nchw [b input-h weight-h bias-h params]
    "NCHW GroupNorm with optional affine parameters; returns a new handle.")
  (-upsample-nearest2d [b input-h params]
    "Integer nearest-neighbor NCHW upsampling; returns a new handle.")
  (-cat [b input-handles params]
    "Concatenate contiguous tensors along an arbitrary axis; returns a new handle."))
