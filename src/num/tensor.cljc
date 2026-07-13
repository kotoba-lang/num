(ns num.tensor
  "Phase 1 (ADR-2607051400 §Phase 1) — the N-D tensor layer on top of `num.array`
  and `num.core`.

  DESIGN CHOICE (documented per the task): this EXTENDS `num.array/NDArray`
  rather than introducing a new type. `NDArray` is already `{:backend :handle
  :shape}` where `shape` is an arbitrary-length vector and `num.array/nelems`
  is already `(reduce * 1 shape)` — that is, the existing record and every
  existing constructor/accessor (`from-vec`/`zeros`/`from-fn`/`->vec`) already
  work for rank-0 (scalar, `shape []`) through arbitrary rank with ZERO changes.
  What was actually missing was not a type but a set of SHAPE-AWARE OPERATIONS
  (broadcasting, reshape/transpose/squeeze/unsqueeze, axis reductions, batched
  matmul) that `num.core` never needed for its 1-D/2-D-only ops. So `num.core`
  is left untouched (its ops keep their existing fast, non-broadcasting,
  direct-to-`IBackend` dispatch — no behavior change, no regression risk) and
  every new N-D op lives here in `num.tensor`.

  HOST-MATERIALIZED, NOT DEVICE-NATIVE (documented tradeoff, matches the ADR's
  own phase split): `num.protocol/IBackend` has no notion of strides, gather,
  or scatter — a handle is an opaque flat contiguous buffer. Pushing
  reshape/transpose/broadcast/reduce/batched-matmul into `IBackend` itself
  (so a GPU backend could execute them without a host round-trip) is exactly
  what the ADR assigns to **Phase 2** (\"N-D matmul/broadcast dispatch is new
  WGSL\"). Phase 1's job is CORRECTNESS of the N-D semantics, held to the same
  \"real numbers, not just doesn't-throw\" bar as `num.cpu`'s reference loops —
  so every op here reads operands back to the host via `num.array/->vec`,
  computes with plain `double-array`/`aget`/`aset` loops (the same style
  `num.cpu` already uses, portable to `.cljc`), and re-uploads the result via
  `num.array/from-vec`. This works identically for ANY backend today (it only
  ever calls the existing `->vec`/`from-vec`/`-ewise` seam) and is the natural
  thing for Phase 2 to later short-circuit with direct-to-device kernels.

  `reshape`/`squeeze`/`unsqueeze` are the one exception: for a row-major
  contiguous layout, inserting/removing a size-1 axis or relabeling the shape
  without touching axis ORDER never moves data, so those three are pure
  metadata edits (`assoc :shape`) — zero-copy, zero host round-trip, on any
  backend.

  KNOWN GAP (found live 2026-07-13, ADR-2607131500 Phase 1): every op here
  that DOES round-trip assumes `num.array/->vec` returns an immediate value —
  true for `num.cpu`'s CpuBackend and `num.wgsl-backend`'s synchronous
  WgslBackend, but NOT true for `num.deno-gpu`'s WgslBackendAsync, whose
  `-copy-to-host` returns a JS Promise (WebGPU readback is inherently async —
  see that namespace's docstring). Calling any op below (including the
  `softmax`/`conv2d`/`attention` this phase adds) against a Deno GPU backend
  throws (`[object Promise] is not ISeqable`), confirmed by direct test.
  Most generic operations in this namespace still require that synchronous
  readback. Device-native exceptions now include metadata-only reshapes and the
  UNet path (SiLU, full NCHW convolution, GroupNorm, nearest upsampling, cat)
  through `ITensorBackend`; they are directly verified on Apple M4 Metal
  (`num.deno-gpu-verify`, 24/24).

  PARTIALLY CLOSED (same day): `num.tensor-async` (cljs-only) adds async
  twins of exactly `conv2d`/`attention`'s underlying ops (2-D transpose, 2-D
  last-axis softmax) — verified live on real Metal AND cross-checked against
  this namespace's own CPU-sync `attention` for the same inputs
  (`num.tensor-async-verify`, 4/4). `num.core/matmul` needed no async
  variant at all: `-gemm`/`-alloc` are fully synchronous even on
  WgslBackendAsync (only `-reduce`/`-dot`/`-nrm2`/`-copy-to-host` are async),
  confirmed by the pre-existing `num.deno-gpu-verify` 14/14. The general N-D
  layer here (broadcast-to/reduce-axes/batched matmul, and `sum`/`amax`/
  `amin`/`mean`) remains sync-only/CPU-verified-only — `num.tensor-async`
  deliberately targets only what `comfyui.nodes.toy-diffusion` needs, not a
  full async generalization of this namespace; see that ns's own docstring."
  (:refer-clojure :exclude [cat])
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.protocol :as p]))

;; --- shape / stride helpers ---------------------------------------------------

(defn- array-dtype [a] (or (:dtype a) :f32))

(defn- require-same-dtype! [operation arrays]
  (let [dtypes (set (map array-dtype (remove nil? arrays)))]
    (when-not (= 1 (count dtypes))
      (throw (ex-info (str operation " requires matching dtypes")
                      {:dtypes dtypes})))))

(defn row-major-strides
  "Row-major (C-order) strides for `shape`: stride[i] = product of shape[i+1..]."
  [shape]
  (let [r (count shape)
        s (long-array (max r 1))]
    (loop [i (dec r) acc 1]
      (if (< i 0)
        (vec (take r (seq s)))
        (do (aset s i acc)
            (recur (dec i) (* acc (long (nth shape i)))))))))

(defn- unravel
  "Linear index `idx` (row-major over `shape`) → per-axis index vector."
  [idx shape strides]
  (mapv (fn [sh st] (if (zero? (long st)) 0 (mod (quot idx st) sh))) shape strides))

(defn- ravel
  "Per-axis index vector → linear offset using `strides` (which may contain 0
  for a broadcast axis, making that axis contribute nothing)."
  [multi-idx strides]
  (reduce + 0 (map * multi-idx strides)))

(defn- normalize-axes
  "`axes` (an int, a negative int, or a collection of either) → a #{} of
  non-negative axis indices valid for a `rank`-dimensional shape."
  [rank axes]
  (let [axes (if (sequential? axes) axes [axes])]
    (set (map (fn [ax]
                (let [ax (long ax)
                      ax (if (neg? ax) (+ rank ax) ax)]
                  (when-not (< -1 ax rank)
                    (throw (ex-info "num.tensor: axis out of range"
                                     {:axis ax :rank rank})))
                  ax))
              axes))))

;; --- broadcasting (NumPy-style) ------------------------------------------------

(defn broadcast-shapes
  "NumPy broadcast rule for two shapes: align from the TRAILING dimension
  (shorter shape is padded on the left with 1s), a size-1 dim stretches to
  match the other operand's size on that axis, and any other size mismatch is
  an error. Returns the resulting shape."
  [s1 s2]
  (let [r (max (count s1) (count s2))
        pad (fn [s] (into (vec (repeat (- r (count s)) 1)) s))
        p1 (pad s1) p2 (pad s2)]
    (mapv (fn [a b]
            (let [a (long a) b (long b)]
              (cond
                (= a b) a
                (= a 1) b
                (= b 1) a
                :else (throw (ex-info "num.tensor/broadcast-shapes: incompatible shapes"
                                       {:shape-1 s1 :shape-2 s2})))))
          p1 p2)))

(defn broadcast-to
  "Return an NDArray with `a`'s data broadcast (NumPy-style) up to
  `target-shape`. Identity (no copy, no host round-trip) when the shapes
  already match. Otherwise materializes a new backend buffer of `target-shape`
  by re-reading `a`'s elements with broadcast (size-1 → stride 0) axes."
  [a target-shape]
  (let [shape (:shape a) target-shape (vec target-shape)]
    (if (= shape target-shape)
      a
      (let [r (count target-shape)
            padded (into (vec (repeat (- r (count shape)) 1)) shape)
            _ (doseq [[s t] (map vector padded target-shape)]
                (when-not (or (= (long s) (long t)) (= (long s) 1))
                  (throw (ex-info "num.tensor/broadcast-to: shape is not broadcastable to target"
                                   {:shape shape :target target-shape}))))
            padded-strides (row-major-strides padded)
            eff-strides (mapv (fn [s st] (if (= (long s) 1) 0 st)) padded padded-strides)
            out-strides (row-major-strides target-shape)
            xs (double-array (arr/->vec a))
            n (arr/nelems target-shape)
            out (double-array n)]
        (dotimes [oi n]
          (aset out oi (aget xs (ravel (unravel oi target-shape out-strides) eff-strides))))
        (arr/from-vec (:backend a) (vec out) target-shape (array-dtype a))))))

(defn- ewise-bc
  "Broadcasting elementwise dispatch: broadcast both operands to their common
  shape (zero-cost when shapes already match — see `broadcast-to`), then
  dispatch the (now equal-shape/equal-size) op straight through `IBackend`,
  exactly like `num.core`'s non-broadcasting `ewise`."
  [op x y]
  (let [shape (broadcast-shapes (:shape x) (:shape y))
        x' (broadcast-to x shape)
        y' (broadcast-to y shape)
        _ (require-same-dtype! "num.tensor elementwise" [x' y'])]
    ((case op :add nm/add :sub nm/sub :mul nm/mul :div nm/div) x' y')))

(defn add "Broadcasting x + y." [x y] (ewise-bc :add x y))
(defn sub "Broadcasting x - y." [x y] (ewise-bc :sub x y))
(defn mul "Broadcasting elementwise x * y (Hadamard, not matmul)." [x y] (ewise-bc :mul x y))
(defn div "Broadcasting elementwise x / y." [x y] (ewise-bc :div x y))

;; --- reshape / transpose / squeeze / unsqueeze --------------------------------

(defn reshape
  "Reinterpret `a`'s (row-major, contiguous) data as `new-shape`. Zero-copy —
  only `:shape` changes. Errors if the element count would change."
  [a new-shape]
  (let [new-shape (vec new-shape)
        old-n (arr/nelems (:shape a)) new-n (arr/nelems new-shape)]
    (when-not (= old-n new-n)
      (throw (ex-info "num.tensor/reshape: element count mismatch"
                       {:from-shape (:shape a) :from-n old-n
                        :to-shape new-shape :to-n new-n})))
    (assoc a :shape new-shape)))

(defn squeeze
  "Remove size-1 axes. With no `axis`, removes every size-1 axis. With an
  `axis` (possibly negative), removes only that axis — error if its size
  isn't 1. Zero-copy."
  ([a] (assoc a :shape (vec (remove #(= (long %) 1) (:shape a)))))
  ([a axis]
   (let [shape (:shape a) r (count shape)
         axis (long (if (neg? axis) (+ r axis) axis))]
     (when-not (< -1 axis r)
       (throw (ex-info "num.tensor/squeeze: axis out of range" {:shape shape :axis axis})))
     (when-not (= 1 (long (nth shape axis)))
       (throw (ex-info "num.tensor/squeeze: axis size is not 1" {:shape shape :axis axis})))
     (assoc a :shape (vec (keep-indexed (fn [i s] (when (not= i axis) s)) shape))))))

(defn unsqueeze
  "Insert a size-1 axis at `axis` (0..rank, negative counts from the end +1,
  numpy `expand_dims` convention). Zero-copy."
  [a axis]
  (let [shape (:shape a) r (count shape)
        axis (long (if (neg? axis) (+ r axis 1) axis))]
    (when-not (<= 0 axis r)
      (throw (ex-info "num.tensor/unsqueeze: axis out of range" {:shape shape :axis axis})))
    (assoc a :shape (vec (concat (subvec shape 0 axis) [1] (subvec shape axis))))))

(defn transpose
  "Permute axes. With no `perm`, reverses all axes (NumPy `.T` convention —
  full reversal, e.g. swaps rows/cols for a 2-D matrix). `perm` (when given)
  must be a permutation of `0..rank-1`; `out.shape[i] = in.shape[perm[i]]`.
  Unlike reshape/squeeze/unsqueeze this DOES move data (row-major layout
  changes under a non-trivial permutation), so it materializes a new buffer."
  ([a] (transpose a (vec (reverse (range (count (:shape a)))))))
  ([a perm]
   (let [shape (:shape a) r (count shape) perm (vec perm)]
     (when-not (= r (count perm))
       (throw (ex-info "num.tensor/transpose: perm length must equal rank"
                        {:shape shape :perm perm})))
     (when-not (= (set perm) (set (range r)))
       (throw (ex-info "num.tensor/transpose: perm must be a permutation of 0..rank-1"
                        {:shape shape :perm perm})))
     (let [out-shape (mapv shape perm)
           in-strides (row-major-strides shape)
           out-strides (row-major-strides out-shape)
           xs (double-array (arr/->vec a))
           n (arr/nelems out-shape)
           out (double-array n)]
       (dotimes [oi n]
         (let [out-idx (unravel oi out-shape out-strides)
               ;; out axis i reads from in axis perm[i]: in-idx[perm[i]] = out-idx[i]
               in-idx (reduce (fn [v [i p]] (assoc v p (nth out-idx i)))
                               (vec (repeat r 0))
                               (map-indexed vector perm))]
           (aset out oi (aget xs (ravel in-idx in-strides)))))
       (arr/from-vec (:backend a) (vec out) out-shape)))))

;; --- axis-parameterized reductions ---------------------------------------------

(defn- reduce-axes
  "Host-computed reduction over `axes` (see `normalize-axes`). `op` ∈
  #{:sum :max :min :mean}."
  [a axes keepdims? op]
  (let [shape (:shape a) r (count shape)
        axes-set (if (nil? axes) (set (range r)) (normalize-axes r axes))
        out-shape-kept (mapv (fn [i s] (if (contains? axes-set i) 1 s)) (range r) shape)
        out-shape-dropped (vec (keep-indexed (fn [i s] (when-not (contains? axes-set i) s)) shape))
        in-strides (row-major-strides shape)
        out-strides-kept (row-major-strides out-shape-kept)
        xs (double-array (arr/->vec a))
        n-in (arr/nelems shape)
        n-out (arr/nelems out-shape-kept)
        init (case op (:sum :mean) 0.0 :max ##-Inf :min ##Inf)
        acc (double-array n-out init)
        cnt (long-array n-out 0)]
    (dotimes [ii n-in]
      (let [in-idx (unravel ii shape in-strides)
            out-idx (map-indexed (fn [i v] (if (contains? axes-set i) 0 v)) in-idx)
            oi (ravel out-idx out-strides-kept)
            v (aget xs ii)]
        (case op
          :sum (aset acc oi (+ (aget acc oi) v))
          :mean (do (aset acc oi (+ (aget acc oi) v))
                    (aset cnt oi (inc (aget cnt oi))))
          :max (aset acc oi (max (aget acc oi) v))
          :min (aset acc oi (min (aget acc oi) v)))))
    (when (= op :mean)
      (dotimes [oi n-out] (aset acc oi (/ (aget acc oi) (aget cnt oi)))))
    (arr/from-vec (:backend a) (vec acc) (if keepdims? out-shape-kept out-shape-dropped))))

(defn sum
  "Σ along `axes` (an axis, a collection of axes, or nil = all axes) → NDArray.
  `opts` is `{:keepdims? bool}` (default false, NumPy/PyTorch convention)."
  ([a] (sum a nil {}))
  ([a axes] (sum a axes {}))
  ([a axes {:keys [keepdims?] :or {keepdims? false}}] (reduce-axes a axes keepdims? :sum)))

(defn amax
  "Max along `axes` (nil = all axes) → NDArray."
  ([a] (amax a nil {}))
  ([a axes] (amax a axes {}))
  ([a axes {:keys [keepdims?] :or {keepdims? false}}] (reduce-axes a axes keepdims? :max)))

(defn amin
  "Min along `axes` (nil = all axes) → NDArray."
  ([a] (amin a nil {}))
  ([a axes] (amin a axes {}))
  ([a axes {:keys [keepdims?] :or {keepdims? false}}] (reduce-axes a axes keepdims? :min)))

(defn mean
  "Arithmetic mean along `axes` (nil = all axes) → NDArray."
  ([a] (mean a nil {}))
  ([a axes] (mean a axes {}))
  ([a axes {:keys [keepdims?] :or {keepdims? false}}] (reduce-axes a axes keepdims? :mean)))

;; --- batched (N-D) matmul -------------------------------------------------------

(defn matmul
  "Batched matmul, generalizing `num.core/matmul` (2-D only): the LAST TWO dims
  of each operand are the matrix dims (`A: [...,m,k]`, `B: [...,k,n]` →
  `[...,m,n]`); any LEADING (batch) dims broadcast NumPy-style, same rule as
  `broadcast-shapes`. A plain 2-D × 2-D call (no batch dims) reduces to the
  same result as `num.core/matmul`."
  [A B]
  (let [sa (:shape A) sb (:shape B) ra (count sa) rb (count sb)]
    (when-not (and (>= ra 2) (>= rb 2))
      (throw (ex-info "num.tensor/matmul: both operands need rank >= 2" {:shape-a sa :shape-b sb})))
    (let [m (long (nth sa (- ra 2))) ka (long (nth sa (dec ra)))
          kb (long (nth sb (- rb 2))) n (long (nth sb (dec rb)))]
      (when-not (= ka kb)
        (throw (ex-info "num.tensor/matmul: inner dimensions must match"
                         {:shape-a sa :shape-b sb})))
      (let [batch-a (vec (drop-last 2 sa))
            batch-b (vec (drop-last 2 sb))
            batch-shape (broadcast-shapes batch-a batch-b)
            nb (long (arr/nelems batch-shape))
            A' (broadcast-to A (into batch-shape [m ka]))
            B' (broadcast-to B (into batch-shape [ka n]))
            xa (double-array (arr/->vec A'))
            xb (double-array (arr/->vec B'))
            out (double-array (* nb m n))]
        (dotimes [bi nb]
          (let [a-off (* bi m ka) b-off (* bi ka n) c-off (* bi m n)]
            (dotimes [i m]
              (dotimes [j n]
                (let [s (loop [l 0 s 0.0]
                          (if (< l ka)
                            (recur (inc l)
                                   (+ s (* (aget xa (+ a-off (* i ka) l))
                                           (aget xb (+ b-off (* l n) j)))))
                            s))]
                  (aset out (+ c-off (* i n) j) s))))))
        (arr/from-vec (:backend A) (vec out) (into batch-shape [m n]))))))

;; --- softmax (ADR-2607131500 Phase 1) ------------------------------------------
;; Everything below reuses ONLY the primitives already real above (amax/sub/sum/
;; div, all real ops going through IBackend, not new host-only shortcuts) plus
;; num.core's exp (the one genuinely-new device primitive this phase adds — see
;; num.protocol/-ewise1, verified on real Metal via num.deno-gpu-verify).

(defn softmax
  "Numerically-stable softmax along `axis` (default: last axis) — subtract the
  per-axis max before exponentiating so large inputs don't overflow, matching
  PyTorch's `nn.functional.softmax` semantics. `exp(x - max) / sum(exp(x - max))`."
  ([a] (softmax a (dec (count (:shape a)))))
  ([a axis]
   (let [mx (amax a axis {:keepdims? true})
         shifted (sub a mx)
         ex (nm/exp shifted)
         s (sum ex axis {:keepdims? true})]
     (div ex s))))

;; --- conv2d (ADR-2607131500 Phase 1) --------------------------------------------

(defn conv2d
  "2-D 'valid' convolution (no padding, stride 1), SINGLE channel, SINGLE
  image: `a` is `[H W]`, `kernel` is `[kh kw]` → NDArray `[H-kh+1, W-kw+1]`.

  im2col + matmul: patch extraction (below) is the one genuinely new op this
  adds, and it is a host round-trip like every other op in this namespace —
  the actual multiply-accumulate FLOPs run through `matmul` (already verified
  real on Metal). Multi-channel / batched / padded / strided conv is real conv
  but a materially bigger op — deliberately NOT attempted here; see
  ADR-2607131500 for the explicit Phase 1 scope fence."
  [a kernel]
  (let [[H W] (:shape a) [kh kw] (:shape kernel)
        oh (inc (- (long H) (long kh))) ow (inc (- (long W) (long kw)))]
    (when (or (< oh 1) (< ow 1))
      (throw (ex-info "num.tensor/conv2d: kernel larger than input"
                       {:input [H W] :kernel [kh kw]})))
    (let [xs (double-array (arr/->vec a))
          W (long W) kh (long kh) kw (long kw)
          patches (double-array (* oh ow kh kw))]
      (dotimes [oi oh]
        (dotimes [oj ow]
          (dotimes [ki kh]
            (dotimes [kj kw]
              (aset patches (+ (* (+ (* oi ow) oj) kh kw) (* ki kw) kj)
                    (aget xs (+ (* (+ oi ki) W) (+ oj kj))))))))
      (let [P (arr/from-vec (:backend a) (vec patches) [(* oh ow) (* kh kw)])
            kflat (reshape kernel [(* kh kw) 1])]
        (reshape (matmul P kflat) [oh ow])))))

;; --- conv2d-mc (multi-channel, 2026-07-13 "raise the maturity" loop) -----------

(defn conv2d-mc
  "Multi-channel 2-D 'valid' convolution (no padding, stride 1), SINGLE
  image (no batch dim): `a` is `[C_in H W]`, `kernel` is
  `[C_out C_in kh kw]` → NDArray `[C_out H-kh+1 W-kw+1]`.

  Generalizes `conv2d` (single-channel) the same im2col+matmul way: each
  output spatial position's receptive field is flattened ACROSS ALL input
  channels into one patches-matrix row; the kernel flattens per output
  channel the same way (`[C_out C_in kh kw]` reshaped to
  `[C_out C_in*kh*kw]` is zero-copy — the layout already matches row-major).
  The actual multiply-accumulate still runs through `matmul` (already
  verified real on Metal). Real diffusion UNets need MANY channels
  (a first conv is typically 3-4 in, 320+ out) — `conv2d`'s single-channel
  restriction was the most toy-like limit from ADR-2607131500 Phase 1; this
  closes it. Still NOT attempted here: batching (multiple images) and
  padding/stride — `conv2d-mc` is one image, valid-only, stride 1, same as
  `conv2d`'s own remaining scope fence."
  [a kernel]
  (let [[Cin H W] (:shape a) [Cout Cin2 kh kw] (:shape kernel)]
    (when (not= (long Cin) (long Cin2))
      (throw (ex-info "num.tensor/conv2d-mc: kernel in-channels ≠ input channels"
                       {:input-channels Cin :kernel-in-channels Cin2})))
    (let [oh (inc (- (long H) (long kh))) ow (inc (- (long W) (long kw)))]
      (when (or (< oh 1) (< ow 1))
        (throw (ex-info "num.tensor/conv2d-mc: kernel larger than input"
                         {:input [H W] :kernel [kh kw]})))
      (let [xs (double-array (arr/->vec a))
            Cin (long Cin) H (long H) W (long W) kh (long kh) kw (long kw)
            patch-size (* Cin kh kw)
            patches (double-array (* oh ow patch-size))]
        (dotimes [oi oh]
          (dotimes [oj ow]
            (let [row (+ (* oi ow) oj)]
              (dotimes [c Cin]
                (dotimes [ki kh]
                  (dotimes [kj kw]
                    (aset patches (+ (* row patch-size) (* c kh kw) (* ki kw) kj)
                          (aget xs (+ (* c H W) (* (+ oi ki) W) (+ oj kj))))))))))
        (let [P (arr/from-vec (:backend a) (vec patches) [(* oh ow) patch-size])
              Kflat-T (transpose (reshape kernel [Cout patch-size]))  ; [patch-size Cout]
              out (matmul P Kflat-T)]                                ; [oh*ow Cout]
          (reshape (transpose out) [Cout oh ow]))))))

(defn- pair-option [option-name value]
  (let [pair (if (sequential? value) (vec value) [value value])]
    (when-not (= 2 (count pair))
      (throw (ex-info (str "num.tensor/conv2d-nchw: " (name option-name) " must be a scalar or pair")
                      {:option option-name :value value})))
    (mapv long pair)))

(defn conv2d-nchw
  "PyTorch-compatible NCHW cross-correlation.

  `input` is `[N C_in H W]`, `weight` is
  `[C_out C_in/groups kH kW]`, and optional `bias` is `[C_out]`.
  Options: `:stride`, `:padding`, and `:dilation` are a scalar or `[h w]`;
  `:groups` defaults to 1. Returns `[N C_out outH outW]` using PyTorch's
  floor output-size rule. Despite the conventional name, this is
  cross-correlation (kernel is not flipped), matching torch/ComfyUI weights.

  This closes the shape/semantics boundary needed by real UNets: batching,
  input/output channels, same-padding, downsampling stride, dilation, grouped
  and depthwise convolution, and bias. Backends implementing
  `num.protocol/ITensorBackend` execute it device-native; other backends use
  the portable host oracle below."
  ([input weight] (conv2d-nchw input weight nil {}))
  ([input weight bias] (conv2d-nchw input weight bias {}))
  ([input weight bias {:keys [stride padding dilation groups]
                       :or {stride 1 padding 0 dilation 1 groups 1}}]
   (require-same-dtype! "num.tensor/conv2d-nchw" [input weight bias])
   (let [input-shape (:shape input)
         weight-shape (:shape weight)]
     (when-not (= 4 (count input-shape))
       (throw (ex-info "num.tensor/conv2d-nchw: input must have shape [N C H W]"
                       {:shape input-shape})))
     (when-not (= 4 (count weight-shape))
       (throw (ex-info "num.tensor/conv2d-nchw: weight must have shape [Cout Cin/groups kH kW]"
                       {:shape weight-shape})))
     (let [[N Cin H W] (mapv long input-shape)
           [Cout Cin-per-group kh kw] (mapv long weight-shape)
           [sh sw] (pair-option :stride stride)
           [ph pw] (pair-option :padding padding)
           [dh dw] (pair-option :dilation dilation)
           groups (long groups)]
       (when-not (and (pos? sh) (pos? sw) (pos? dh) (pos? dw) (pos? groups))
         (throw (ex-info "num.tensor/conv2d-nchw: stride, dilation, and groups must be positive"
                         {:stride [sh sw] :dilation [dh dw] :groups groups})))
       (when (or (neg? ph) (neg? pw))
         (throw (ex-info "num.tensor/conv2d-nchw: padding must be non-negative"
                         {:padding [ph pw]})))
       (when-not (and (zero? (mod Cin groups))
                      (zero? (mod Cout groups))
                      (= Cin-per-group (quot Cin groups)))
         (throw (ex-info "num.tensor/conv2d-nchw: channels are incompatible with groups"
                         {:input-channels Cin :output-channels Cout
                          :weight-input-channels Cin-per-group :groups groups})))
       (when (and bias (not= [Cout] (:shape bias)))
         (throw (ex-info "num.tensor/conv2d-nchw: bias must have shape [Cout]"
                         {:expected [Cout] :actual (:shape bias)})))
       (let [effective-kh (inc (* dh (dec kh)))
             effective-kw (inc (* dw (dec kw)))
             _ (when (or (< (+ H (* 2 ph)) effective-kh)
                         (< (+ W (* 2 pw)) effective-kw))
                 (throw (ex-info "num.tensor/conv2d-nchw: effective kernel larger than padded input"
                                 {:input [H W] :kernel [kh kw] :padding [ph pw]
                                  :dilation [dh dw]})))
             oh (inc (quot (- (+ H (* 2 ph)) effective-kh) sh))
             ow (inc (quot (- (+ W (* 2 pw)) effective-kw) sw))
             backend (:backend input)
             params {:n N :cin Cin :h H :width W :cout Cout
                     :cin-group Cin-per-group :kh kh :kw kw :oh oh :ow ow
                     :sh sh :sw sw :ph ph :pw pw :dh dh :dw dw :groups groups}]
         (cond
           (and (not= :f32 (array-dtype input))
                (satisfies? p/IDTypeTensorOps backend))
           (assoc (arr/->NDArray backend
                                  (p/-conv2d-nchw-dtype
                                   backend (:handle input) (:handle weight)
                                   (when bias (:handle bias)) params (array-dtype input))
                                  [N Cout oh ow])
                  :dtype (array-dtype input))

           (and (= :f32 (array-dtype input))
                (satisfies? p/ITensorBackend backend))
           (assoc (arr/->NDArray backend
                          (p/-conv2d-nchw backend (:handle input) (:handle weight)
                                          (when bias (:handle bias)) params)
                          [N Cout oh ow]) :dtype :f32)
           :else
           (let [xs (double-array (arr/->vec input))
                 ws (double-array (arr/->vec weight))
                 bs (when bias (double-array (arr/->vec bias)))
                 out (double-array (* N Cout oh ow))
                 outputs-per-group (quot Cout groups)]
             (dotimes [n N]
               (dotimes [oc Cout]
                 (let [group (quot oc outputs-per-group)
                       input-channel-base (* group Cin-per-group)]
                   (dotimes [oi oh]
                     (dotimes [oj ow]
                       (let [sum
                             (loop [icg 0 sum (if bs (aget bs oc) 0.0)]
                               (if (< icg Cin-per-group)
                                 (let [ic (+ input-channel-base icg)
                                       sum
                                       (loop [ki 0 sum sum]
                                         (if (< ki kh)
                                           (let [ih (+ (- (* oi sh) ph) (* ki dh))
                                                 sum
                                                 (if (or (neg? ih) (>= ih H))
                                                   sum
                                                   (loop [kj 0 sum sum]
                                                     (if (< kj kw)
                                                       (let [iw (+ (- (* oj sw) pw) (* kj dw))]
                                                         (recur (inc kj)
                                                                (if (or (neg? iw) (>= iw W))
                                                                  sum
                                                                  (+ sum
                                                                     (* (aget xs (+ (* n Cin H W)
                                                                                    (* ic H W)
                                                                                    (* ih W) iw))
                                                                        (aget ws (+ (* oc Cin-per-group kh kw)
                                                                                    (* icg kh kw)
                                                                                    (* ki kw) kj)))))))
                                                       sum)))]
                                             (recur (inc ki) sum))
                                           sum))]
                                   (recur (inc icg) sum))
                                 sum))]
                         (aset out (+ (* n Cout oh ow) (* oc oh ow) (* oi ow) oj)
                               sum)))))))
             (arr/from-vec backend (vec out) [N Cout oh ow]
                           (array-dtype input)))))))))

;; --- UNet tensor building blocks -----------------------------------------------

(defn silu
  "Elementwise SiLU/Swish: `x * sigmoid(x)`, dispatched as one backend unary
  kernel (device-native on WGSL/Metal; no host readback)."
  [input]
  (nm/silu input))

(defn cat
  "Concatenate equal-rank tensors along `axis` (PyTorch `torch.cat` shape
  semantics). All non-concatenated dimensions must match. ITensorBackend
  implementations dispatch device-to-device slice copies."
  [tensors axis]
  (let [tensors (vec tensors)]
    (when (empty? tensors)
      (throw (ex-info "num.tensor/cat requires at least one tensor" {})))
    (let [first-shape (:shape (first tensors))
          rank (count first-shape)
          axis (long (if (neg? axis) (+ rank axis) axis))]
      (require-same-dtype! "num.tensor/cat" tensors)
      (when-not (< -1 axis rank)
        (throw (ex-info "num.tensor/cat axis out of range"
                        {:axis axis :rank rank})))
      (doseq [tensor tensors]
        (when-not (and (= rank (count (:shape tensor)))
                       (every? true?
                               (map-indexed
                                (fn [i size]
                                  (or (= i axis) (= size (nth first-shape i))))
                                (:shape tensor))))
          (throw (ex-info "num.tensor/cat shapes differ outside axis"
                          {:shapes (mapv :shape tensors) :axis axis}))))
      (let [out-shape (assoc first-shape axis
                             (reduce + (map #(long (nth (:shape %) axis)) tensors)))
            outer (long (arr/nelems (subvec first-shape 0 axis)))
            inner (long (arr/nelems (subvec first-shape (inc axis))))
            axis-sizes (mapv #(long (nth (:shape %) axis)) tensors)
            backend (:backend (first tensors))]
        (if (and (= :f32 (array-dtype (first tensors)))
                 (satisfies? p/ITensorBackend backend)
                 (every? #(= backend (:backend %)) tensors))
          (let [offsets (butlast (reductions + 0 axis-sizes))
                inputs (mapv (fn [tensor axis-size axis-offset]
                               {:total (arr/nelems (:shape tensor))
                                :block (* axis-size inner)
                                :axis-offset (* axis-offset inner)})
                             tensors axis-sizes offsets)
                params {:total-output (arr/nelems out-shape)
                        :output-block (* (nth out-shape axis) inner)
                        :inputs inputs}]
            (assoc (arr/->NDArray backend
                           (p/-cat backend (mapv :handle tensors) params)
                           out-shape) :dtype :f32))
          (let [out (double-array (arr/nelems out-shape))
                sources (mapv #(double-array (arr/->vec %)) tensors)]
            (dotimes [outer-index outer]
              (loop [tensor-index 0 axis-offset 0]
                (when (< tensor-index (count tensors))
                  (let [axis-size (nth axis-sizes tensor-index)
                        ^doubles source (nth sources tensor-index)
                        count (* axis-size inner)
                        source-base (* outer-index count)
                        output-base (+ (* outer-index (nth out-shape axis) inner)
                                       (* axis-offset inner))]
                    (dotimes [i count]
                      (aset out (+ output-base i) (aget source (+ source-base i))))
                    (recur (inc tensor-index) (+ axis-offset axis-size))))))
            (arr/from-vec backend (vec out) out-shape
                          (array-dtype (first tensors)))))))))

(defn group-norm-nchw
  "PyTorch-compatible GroupNorm for `[N C H W]`. Variance is biased
  (`unbiased=false`), as in `torch.nn.GroupNorm`. Optional affine `weight` and
  `bias` have shape `[C]`. ITensorBackend implementations run device-native;
  the portable fallback is the CPU oracle below."
  ([input num-groups] (group-norm-nchw input num-groups nil nil 1.0e-5))
  ([input num-groups weight bias eps]
   (require-same-dtype! "num.tensor/group-norm-nchw" [input weight bias])
   (let [[N C H W :as shape] (:shape input)
         groups (long num-groups)]
     (when-not (and (= 4 (count shape)) (pos? groups) (zero? (mod (long C) groups))
                    (<= 0.0 eps))
       (throw (ex-info "num.tensor/group-norm-nchw requires [N C H W] and groups dividing C"
                       {:shape shape :groups groups :eps eps})))
     (when (and weight (not= [(long C)] (:shape weight)))
       (throw (ex-info "group norm weight must have shape [C]" {:shape (:shape weight)})))
     (when (and bias (not= [(long C)] (:shape bias)))
       (throw (ex-info "group norm bias must have shape [C]" {:shape (:shape bias)})))
     (let [N (long N) C (long C) H (long H) W (long W)
           channels-per-group (quot C groups)
           group-size (* channels-per-group H W)
           backend (:backend input)
           params {:n N :c C :h H :width W :groups groups
                   :channels-group channels-per-group :group-size group-size
                   :eps eps}]
       (cond
         (and (not= :f32 (array-dtype input))
              (satisfies? p/IDTypeTensorOps backend))
         (assoc (arr/->NDArray backend
                                (p/-group-norm-nchw-dtype
                                 backend (:handle input)
                                 (when weight (:handle weight))
                                 (when bias (:handle bias)) params (array-dtype input))
                                [N C H W])
                :dtype (array-dtype input))

         (and (= :f32 (array-dtype input))
              (satisfies? p/ITensorBackend backend))
         (assoc (arr/->NDArray backend
                        (p/-group-norm-nchw backend (:handle input)
                                            (when weight (:handle weight))
                                            (when bias (:handle bias)) params)
                        [N C H W]) :dtype :f32)
         :else
         (let [xs (double-array (arr/->vec input))
               ws (when weight (double-array (arr/->vec weight)))
               bs (when bias (double-array (arr/->vec bias)))
               out (double-array (* N C H W))]
           (dotimes [n N]
             (dotimes [group groups]
               (let [base (+ (* n C H W) (* group group-size))
                     mean (/ (loop [i 0 sum 0.0]
                               (if (< i group-size)
                                 (recur (inc i) (+ sum (aget xs (+ base i))))
                                 sum))
                             group-size)
                     variance (/ (loop [i 0 sum 0.0]
                                   (if (< i group-size)
                                     (let [d (- (aget xs (+ base i)) mean)]
                                       (recur (inc i) (+ sum (* d d))))
                                     sum))
                                 group-size)
                     inv-std (/ 1.0 (Math/sqrt (+ variance eps)))]
                 (dotimes [i group-size]
                   (let [channel (+ (* group channels-per-group)
                                    (quot i (* H W)))
                         normalized (* (- (aget xs (+ base i)) mean) inv-std)
                         value (+ (* normalized (if ws (aget ws channel) 1.0))
                                  (if bs (aget bs channel) 0.0))]
                     (aset out (+ base i) value))))))
           (arr/from-vec backend (vec out) [N C H W]
                         (array-dtype input))))))))

(defn upsample-nearest2d
  "Nearest-neighbor NCHW upsampling by an integer scalar or `[scale-h scale-w]`.
  ITensorBackend implementations execute it device-native."
  [input scale-factor]
  (let [[N C H W :as shape] (:shape input)
        [scale-h scale-w] (pair-option :scale-factor scale-factor)]
    (when-not (and (= 4 (count shape)) (pos? scale-h) (pos? scale-w))
      (throw (ex-info "num.tensor/upsample-nearest2d requires NCHW and positive scale"
                      {:shape shape :scale-factor [scale-h scale-w]})))
    (let [N (long N) C (long C) H (long H) W (long W)
          oh (* H scale-h) ow (* W scale-w)
          backend (:backend input)
          params {:n N :c C :h H :width W :oh oh :ow ow
                  :scale-h scale-h :scale-w scale-w}]
      (if (and (= :f32 (array-dtype input))
               (satisfies? p/ITensorBackend backend))
        (assoc (arr/->NDArray backend
                       (p/-upsample-nearest2d backend (:handle input) params)
                       [N C oh ow]) :dtype :f32)
        (let [xs (double-array (arr/->vec input))
              out (double-array (* N C oh ow))]
          (dotimes [n N]
            (dotimes [c C]
              (dotimes [oi oh]
                (dotimes [oj ow]
                  (aset out (+ (* n C oh ow) (* c oh ow) (* oi ow) oj)
                        (aget xs (+ (* n C H W) (* c H W)
                                    (* (quot oi scale-h) W) (quot oj scale-w))))))))
          (arr/from-vec backend (vec out) [N C oh ow]
                        (array-dtype input)))))))

;; --- attention (ADR-2607131500 Phase 1) -----------------------------------------

(defn attention
  "Single-head scaled dot-product attention: `softmax(QK^T / √d) · V`. `Q` is
  `[seqQ d]`, `K`/`V` are `[seqK d]` (cross-attention: seqQ may differ from
  seqK). No batching, no multi-head split, no causal/padding mask — the
  honest minimal form; see ADR-2607131500 for what's deliberately deferred."
  [Q K V]
  (let [d (long (last (:shape Q)))
        scores (matmul Q (transpose K))
        scaled (nm/scal! (/ 1.0 (Math/sqrt d)) scores)
        weights (softmax scaled)]
    (matmul weights V)))

;; --- multi-head attention (2026-07-13 "raise the maturity" loop) ---------------

(defn multi-head-attention
  "Multi-head scaled dot-product attention — `attention` generalized the way
  a real transformer/UNet attention block actually needs it (single-head was
  ADR-2607131500 Phase 1's honest-minimal simplification). `Q` is
  `[seqQ d_model]`, `K`/`V` are `[seqK d_model]` (cross-attention: seqQ may
  differ from seqK) — UNLIKE `attention`, `V`'s last dim MUST equal Q/K's
  `d_model` here (the standard transformer convention: V gets split into
  heads the same way Q/K do, so heads can concatenate back to `d_model`;
  `attention` never splits into heads so it tolerates a different `d_v`).
  `num-heads` must evenly divide `d_model`; `num-heads=1`
  reduces to EXACTLY `attention`'s own result (see
  `test/num/tensor_test.cljc`, verified against `attention` directly, not
  just internally consistent). No batching (a single Q/K/V triple, not a
  batch of them) and no causal/padding mask — still deferred, same as
  `attention`'s own scope fence.

  Built from already-real, already-verified primitives only — no new WGSL
  kernel, no new host-round-trip primitive: `reshape`/`transpose` split
  `[seq d_model]` into `[num-heads seq d_head]` and merge back, `matmul`
  handles the per-head matmuls directly (already batched, see its own
  docstring — no new op needed), `softmax` handles the per-head-per-row
  normalization (last axis, already correct for a 3-D `[h seqQ seqK]`
  scores tensor)."
  [Q K V num-heads]
  (let [[seqQ d-model] (:shape Q) [seqK _] (:shape K) h (long num-heads)]
    (when-not (zero? (mod (long d-model) h))
      (throw (ex-info "num.tensor/multi-head-attention: num-heads must evenly divide d_model"
                       {:d-model d-model :num-heads h})))
    (let [d-head (/ (long d-model) h)
          split-heads (fn [x seq-len] (transpose (reshape x [seq-len h d-head]) [1 0 2]))
          Qh (split-heads Q seqQ) Kh (split-heads K seqK) Vh (split-heads V seqK)
          scores (matmul Qh (transpose Kh [0 2 1]))
          scaled (nm/scal! (/ 1.0 (Math/sqrt d-head)) scores)
          weights (softmax scaled)
          heads-out (matmul weights Vh)
          merged (transpose heads-out [1 0 2])]
      (reshape merged [seqQ d-model]))))
