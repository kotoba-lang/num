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
  backend."
  (:require [num.array :as arr]
            [num.protocol :as p]))

;; --- shape / stride helpers ---------------------------------------------------

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
        (arr/from-vec (:backend a) (vec out) target-shape)))))

(defn- ewise-bc
  "Broadcasting elementwise dispatch: broadcast both operands to their common
  shape (zero-cost when shapes already match — see `broadcast-to`), then
  dispatch the (now equal-shape/equal-size) op straight through `IBackend`,
  exactly like `num.core`'s non-broadcasting `ewise`."
  [op x y]
  (let [shape (broadcast-shapes (:shape x) (:shape y))
        x' (broadcast-to x shape)
        y' (broadcast-to y shape)
        b (:backend x')]
    (arr/->NDArray b (p/-ewise b op (:handle x') (:handle y') (arr/nelems shape)) shape)))

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
