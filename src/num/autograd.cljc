(ns num.autograd
  "Minimal reverse-mode automatic differentiation (ADR-2607131500 Phase 1,
  'training'). Deliberately NOT a general training framework — the honest
  minimal form of 'training capability': real, correct gradients (verified
  against numerical/finite-difference gradients, the standard autograd
  correctness check — see test/num/autograd_test.cljc) for exactly the ops a
  tiny linear/relu/linear/softmax(+mse-loss) network needs, matching
  torch-clj's own README example architecture. Most general tensor gradients
  remain host-materialized on GPU backends, but multi-head-attention has a fused
  `ITensorBackend` forward/backward path verified through async Deno WebGPU on
  Apple Metal; its Q/K/V tensors and gradients stay device-resident.

  Design: a small tape-based (Wengert-list) autograd. `with-tape` binds a
  fresh tape; every op below (`matmul*`/`add-bias*`/`relu*`/`softmax*`/
  `mse-loss*`) builds its forward result AND records itself + a
  `backward-fn` closure onto the tape. `backward!` seeds the output's
  gradient and replays the tape in REVERSE (creation) order, each node's
  backward-fn reading its own accumulated upstream gradient and pushing
  gradient contributions into its parents' `:grad` atoms. This is correct
  for the acyclic, single-path sequential graphs a `torch.model/sequential`
  produces — it does NOT handle branching/shared-subgraph reuse (a value
  used by two different downstream ops) or control flow, which a general
  autograd needs and this one deliberately does not attempt."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.protocol :as p]
            [num.tensor :as t]))

(def ^:dynamic *tape* nil)

(defrecord Value [data grad backward-fn parents])

(defn- accumulate! [v contrib]
  (swap! (:grad v) (fn [g] (if g (t/add g contrib) contrib))))

(defn- record! [v]
  (when *tape* (swap! *tape* conj v))
  v)

(defn- node [data parents backward-fn]
  (record! (->Value data (atom nil) backward-fn parents)))

(defn value
  "Wrap `data` (an NDArray) as a graph leaf — a parameter (its `:grad` is
  read after `backward!`) or an input (its `:grad` is simply unused)."
  [data]
  (node data [] (fn [_])))

(defmacro with-tape
  "Run `body` with a fresh tape bound. Returns `[result tape-atom]`."
  [& body]
  `(binding [*tape* (atom [])]
     [(do ~@body) *tape*]))

(defn backward!
  "Seed `v`'s gradient with `seed` (an NDArray matching `v`'s shape — for a
  scalar loss node, a 1-element NDArray) and propagate through `tape` (the
  second element `with-tape` returned) in reverse creation order."
  [v seed tape]
  (accumulate! v seed)
  (doseq [nd (reverse @tape)]
    ((:backward-fn nd) nd)))

;; --- ops -------------------------------------------------------------------

(defn- swap-last-two
  "Transpose the matrix dimensions while preserving any batch dimensions."
  [a]
  (let [rank (count (:shape a))
        perm (vec (concat (range (- rank 2)) [(dec rank) (- rank 2)]))]
    (t/transpose a perm)))

(defn matmul*
  "z = x @ W, including batched matrix multiplication. Gradients transpose
  only the final two (matrix) axes and preserve all leading batch axes."
  [x W]
  (node (t/matmul (:data x) (:data W))
        [x W]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (t/matmul g (swap-last-two (:data W))))
            (accumulate! W (t/matmul (swap-last-two (:data x)) g))))))

(defn add-bias*
  "z = x + b, b broadcast over x's leading (batch) axis. dL/dx = dL/dz;
  dL/db = sum of dL/dz over the batch axis (every batch row's gradient
  contributes to the one shared bias vector)."
  [x b]
  (node (t/add (:data x) (:data b))
        [x b]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x g)
            (accumulate! b (t/sum g 0))))))

(defn- relu-grad
  "Elementwise derivative of relu at `x-data`: 1.0 where x>0, else 0.0 (the
  standard, if technically-debatable-at-exactly-0, convention). A small host
  round-trip — the same documented tradeoff every num.tensor op already
  makes, not a new one."
  [x-data]
  (arr/from-vec (:backend x-data)
                (mapv #(if (pos? %) 1.0 0.0) (arr/->vec x-data))
                (:shape x-data)))

(defn relu*
  [x]
  (node (nm/relu (:data x))
        [x]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (nm/mul g (relu-grad (:data x))))))))

(defn softmax*
  "y = softmax(x) along the last axis. Standard softmax-Jacobian-vector
  product: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j)) (per row)."
  [x]
  (let [y (t/softmax (:data x))]
    (node y
          [x]
          (fn [self]
            (when-let [g @(:grad self)]
              (let [gy (nm/mul g y)
                    s (t/sum gy (dec (count (:shape y))) {:keepdims? true})
                    dx (nm/mul y (t/sub g s))]
                (accumulate! x dx)))))))

(defn mse-loss*
  "L = mean((pred - target)^2) over every element, a scalar (shape `[]`).
  `target` is a plain NDArray, not a Value — nothing needs its gradient.
  dL/dpred = 2*(pred - target) / N."
  [pred target]
  (let [prediction (:data pred)
        backend (:backend prediction)
        shape (:shape prediction)
        count (arr/nelems shape)]
    (when-not (= shape (:shape target))
      (throw (ex-info "num.autograd/mse-loss*: target shape must match prediction"
                      {:prediction shape :target (:shape target)})))
    (if (and (= backend (:backend target))
             (= :f32 (:dtype prediction :f32) (:dtype target :f32))
             (satisfies? p/ITensorBackend backend))
      (node (assoc (arr/->NDArray
                    backend
                    (p/-mse-loss backend (:handle prediction) (:handle target)
                                 {:count count})
                    []) :dtype :f32)
            [pred]
            (fn [self]
              (when-let [g @(:grad self)]
                (accumulate!
                 pred
                 (assoc (arr/->NDArray
                         backend
                         (p/-mse-gradient backend (:handle prediction)
                                          (:handle target) (:handle g)
                                          {:count count})
                         shape) :dtype :f32)))))
      (let [diff (t/sub prediction target)
            n (double count)
            loss-val (if (= :f32 (or (:dtype diff) :f32))
                       (/ (nm/sum (nm/mul diff diff)) n)
                       (/ (reduce + (map #(* % %) (arr/->vec diff))) n))]
        (node (arr/from-vec backend [loss-val] [])
              [pred]
              (fn [self]
                (when-let [g @(:grad self)]
                  (let [scale (* 2.0 (arr/->scalar g) (/ 1.0 n))
                        diff-copy (arr/from-vec backend (arr/->vec diff) shape)]
                    (accumulate! pred (nm/scal! scale diff-copy))))))))))

;; --- conv2d (2026-07-13 "raise the maturity" loop) --------------------------
;; num.tensor/conv2d bundles im2col + matmul + reshape into one opaque
;; function with no seam to hook a backward onto. conv2d* below decomposes
;; the SAME shape convention (x: [H W], kernel: [kh kw], out: [oh ow]) back
;; into that exact im2col+matmul+reshape pipeline, wrapping each step as a
;; differentiable node — matmul* and reshape*'s gradients are already real;
;; im2col*/col2im (below) is the one genuinely new backward formula.

(defn reshape*
  "z = reshape(x, new-shape) (zero-copy, like num.tensor/reshape). dL/dx =
  reshape(dL/dz, x's original shape) — reshape's backward is itself, since
  it never moves data, only relabels shape."
  [x new-shape]
  (let [old-shape (:shape (:data x))]
    (node (t/reshape (:data x) new-shape)
          [x]
          (fn [self]
            (when-let [g @(:grad self)]
              (accumulate! x (t/reshape g old-shape)))))))

(defn- im2col-nd
  "[oh*ow kh*kw] patches NDArray from a `[H W]` NDArray `a` — the exact same
  patch layout `num.tensor/conv2d` uses internally (mirrored here rather
  than reused, since that function has no seam to extract just this step)."
  [a kh kw]
  (let [[H W] (:shape a) kh (long kh) kw (long kw)
        oh (inc (- (long H) kh)) ow (inc (- (long W) kw))
        xs (vec (arr/->vec a))
        W (long W)
        patches (double-array (* oh ow kh kw))]
    (dotimes [oi oh]
      (dotimes [oj ow]
        (dotimes [ki kh]
          (dotimes [kj kw]
            (aset patches (+ (* (+ (* oi ow) oj) kh kw) (* ki kw) kj)
                  (nth xs (+ (* (+ oi ki) W) (+ oj kj))))))))
    (arr/from-vec (:backend a) (vec patches) [(* oh ow) (* kh kw)])))

(defn- col2im-nd
  "Adjoint of `im2col-nd`: scatter-ADD `patches-grad` (`[oh*ow kh*kw]`) back
  to an `in-shape`-shaped (`[H W]`) gradient. A pixel touched by more than
  one sliding window (any interior pixel once `kh`/`kw` > 1) accumulates the
  SUM of every window's contribution — the standard im2col-conv adjoint."
  [patches-grad in-shape kh kw]
  (let [[H W] in-shape kh (long kh) kw (long kw)
        oh (inc (- (long H) kh)) ow (inc (- (long W) kw))
        pg (vec (arr/->vec patches-grad))
        W (long W)
        out (double-array (* (long H) W))]
    (dotimes [oi oh]
      (dotimes [oj ow]
        (dotimes [ki kh]
          (dotimes [kj kw]
            (let [idx (+ (* (+ oi ki) W) (+ oj kj))
                  pidx (+ (* (+ (* oi ow) oj) kh kw) (* ki kw) kj)]
              (aset out idx (+ (aget out idx) (double (nth pg pidx)))))))))
    (arr/from-vec (:backend patches-grad) (vec out) in-shape)))

(defn im2col*
  [x kh kw]
  (node (im2col-nd (:data x) kh kw)
        [x]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (col2im-nd g (:shape (:data x)) kh kw))))))

(defn conv2d*
  "Single-channel 2-D 'valid' convolution with real gradients — same shape
  convention as `num.tensor/conv2d` (`x`: `[H W]`, `kernel`: `[kh kw]`,
  output `[oh ow]`), built from `im2col*` (the one new backward formula) +
  the already-real `matmul*`/`reshape*`."
  [x kernel]
  (let [[kh kw] (:shape (:data kernel))
        kh (long kh) kw (long kw)
        [H W] (:shape (:data x))
        oh (inc (- (long H) kh)) ow (inc (- (long W) kw))
        P (im2col* x kh kw)
        kflat (reshape* kernel [(* kh kw) 1])
        out-flat (matmul* P kflat)]
    (reshape* out-flat [oh ow])))

;; --- attention (2026-07-13 "raise the maturity" loop) -----------------------

(defn transpose*
  "Permute tensor axes with a differentiable transpose. With no permutation,
  reverses all axes like `num.tensor/transpose`. Backward applies the inverse
  permutation, so arbitrary-rank head splitting/merging remains differentiable."
  ([x]
   (transpose* x (vec (reverse (range (count (:shape (:data x))))))))
  ([x perm]
   (let [perm (mapv long perm)
         inverse (reduce-kv (fn [out destination source]
                              (assoc out source destination))
                            (vec (repeat (count perm) 0)) perm)]
     (node (t/transpose (:data x) perm)
           [x]
           (fn [self]
             (when-let [g @(:grad self)]
               (accumulate! x (t/transpose g inverse))))))))

(defn- copy-nd [a] (arr/from-vec (:backend a) (arr/->vec a) (:shape a)))

(defn scale*
  "z = alpha * x (alpha a host scalar constant, not a Value — nothing needs
  its gradient). dL/dx = alpha * dL/dz. Copies before calling num.core/scal!
  (BLAS in-place convention) since `x` — e.g. self-attention's Q=K=V=the
  SAME Value — may still be needed unscaled elsewhere in the graph."
  [alpha x]
  (node (nm/scal! alpha (copy-nd (:data x)))
        [x]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (nm/scal! alpha (copy-nd g)))))))

(defn attention*
  "Single-head scaled dot-product attention with real gradients — same shape
  convention as `num.tensor/attention` (`Q`: `[seqQ d]`, `K`/`V`:
  `[seqK d]`; self-attention passes the SAME Value as all three, which
  `with-tape`/`backward!`'s gradient-accumulation already handles correctly
  — each of Q/K/V's usage sites contributes additively to the one shared
  `:grad` atom). Built from `transpose*`/`scale*` (new) + the already-real
  `matmul*`/`softmax*` — no formula here is independently re-derived, this
  is the identical composition `num.tensor/attention` itself uses."
  [Q K V]
  (let [d (long (last (:shape (:data Q))))
        scores (matmul* Q (transpose* K))
        scaled (scale* (/ 1.0 (Math/sqrt d)) scores)
        weights (softmax* scaled)]
    (matmul* weights V)))

(defn multi-head-attention*
  "Differentiable multi-head scaled dot-product attention matching
  `num.tensor/multi-head-attention`. Q is `[seqQ d-model]`, K/V are
  `[seqK d-model]`; `num-heads` evenly divides d-model."
  [Q K V num-heads]
  (let [[seq-q d-model] (:shape (:data Q))
        [seq-k k-d-model] (:shape (:data K))
        [_ v-d-model] (:shape (:data V))
        heads (long num-heads)]
    (when-not (and (pos? heads)
                   (= d-model k-d-model v-d-model)
                   (zero? (mod (long d-model) heads)))
      (throw (ex-info "num.autograd/multi-head-attention*: incompatible dimensions"
                      {:q-shape (:shape (:data Q))
                       :k-shape (:shape (:data K))
                       :v-shape (:shape (:data V))
                       :num-heads heads})))
    (let [d-head (quot (long d-model) heads)
          q-data (:data Q) k-data (:data K) v-data (:data V)
          backend (:backend q-data)
          fused? (and (= backend (:backend k-data) (:backend v-data))
                      (= :f32 (:dtype q-data :f32)
                         (:dtype k-data :f32) (:dtype v-data :f32))
                      (satisfies? p/ITensorBackend backend))]
      (if fused?
        (node
         (t/multi-head-attention q-data k-data v-data heads)
         [Q K V]
         (fn [self]
           (when-let [g @(:grad self)]
             (let [params {:seq-q seq-q :seq-k seq-k :d-model d-model
                           :heads heads :head-dim d-head}
                   gradients (p/-multi-head-attention-backward
                              backend (:handle q-data) (:handle k-data)
                              (:handle v-data) (:handle g) params)
                   ndarray (fn [handle shape]
                             (assoc (arr/->NDArray backend handle shape)
                                    :dtype :f32))]
               (accumulate! Q (ndarray (:query gradients) [seq-q d-model]))
               (accumulate! K (ndarray (:key gradients) [seq-k d-model]))
               (accumulate! V (ndarray (:value gradients) [seq-k d-model]))))))
        (let [split-heads (fn [x seq-len]
                            (transpose* (reshape* x [seq-len heads d-head]) [1 0 2]))
              qh (split-heads Q seq-q)
              kh (split-heads K seq-k)
              vh (split-heads V seq-k)
              scores (matmul* qh (transpose* kh [0 2 1]))
              weights (softmax* (scale* (/ 1.0 (Math/sqrt d-head)) scores))
              heads-out (matmul* weights vh)]
          (reshape* (transpose* heads-out [1 0 2]) [seq-q d-model]))))))
(defn- pair-option [value]
  (mapv long (if (sequential? value) value [value value])))

(defn conv2d-nchw*
  "Differentiable NCHW convolution matching `num.tensor/conv2d-nchw`.

  `x`, `weight`, and optional `bias` are autograd Values. Backward computes
  gradients for all three and supports the forward op's stride, padding,
  dilation, and groups semantics, including depthwise convolution."
  ([x weight] (conv2d-nchw* x weight nil {}))
  ([x weight bias] (conv2d-nchw* x weight bias {}))
  ([x weight bias {:keys [stride padding dilation groups] :as opts
                   :or {stride 1 padding 0 dilation 1 groups 1}}]
   (let [input-data (:data x)
         weight-data (:data weight)
         bias-data (when bias (:data bias))
         output (t/conv2d-nchw input-data weight-data bias-data opts)
         [N Cin H W] (mapv long (:shape input-data))
         [Cout Cin-per-group kh kw] (mapv long (:shape weight-data))
         [_ _ oh ow] (mapv long (:shape output))
         [sh sw] (pair-option stride)
         [ph pw] (pair-option padding)
         [dh dw] (pair-option dilation)
         groups (long groups)
         outputs-per-group (quot Cout groups)]
     (node output
           (cond-> [x weight] bias (conj bias))
           (fn [self]
             (when-let [g @(:grad self)]
               (let [xs (double-array (arr/->vec input-data))
                     ws (double-array (arr/->vec weight-data))
                     gs (double-array (arr/->vec g))
                     dx (double-array (* N Cin H W))
                     dw-gradient (double-array (* Cout Cin-per-group kh kw))
                     db-gradient (when bias (double-array Cout))]
                 (dotimes [n N]
                   (dotimes [oc Cout]
                     (let [group (quot oc outputs-per-group)
                           input-channel-base (* group Cin-per-group)]
                       (dotimes [oi oh]
                         (dotimes [oj ow]
                           (let [g-index (+ (* n Cout oh ow) (* oc oh ow) (* oi ow) oj)
                                 gradient (aget gs g-index)]
                             (when db-gradient
                               (aset db-gradient oc (+ (aget db-gradient oc) gradient)))
                             (dotimes [icg Cin-per-group]
                               (let [ic (+ input-channel-base icg)]
                                 (dotimes [ki kh]
                                   (let [ih (+ (- (* oi sh) ph) (* ki dh))]
                                     (when (and (<= 0 ih) (< ih H))
                                       (dotimes [kj kw]
                                         (let [iw (+ (- (* oj sw) pw) (* kj dw))]
                                           (when (and (<= 0 iw) (< iw W))
                                             (let [x-index (+ (* n Cin H W) (* ic H W) (* ih W) iw)
                                                   w-index (+ (* oc Cin-per-group kh kw)
                                                              (* icg kh kw) (* ki kw) kj)]
                                               (aset dx x-index
                                                     (+ (aget dx x-index)
                                                        (* gradient (aget ws w-index))))
                                               (aset dw-gradient w-index
                                                     (+ (aget dw-gradient w-index)
                                                        (* gradient (aget xs x-index)))))))))))))))))))
                 (accumulate! x (arr/from-vec (:backend input-data) (vec dx) (:shape input-data)))
                 (accumulate! weight
                              (arr/from-vec (:backend weight-data) (vec dw-gradient)
                                            (:shape weight-data)))
                 (when bias
                   (accumulate! bias
                                (arr/from-vec (:backend bias-data) (vec db-gradient)
                                              (:shape bias-data)))))))))))

;; --- UNet activations / normalization -----------------------------------------

(defn silu*
  "Differentiable SiLU. `d/dx silu(x) = sigmoid(x) *
  (1 + x * (1 - sigmoid(x)))`."
  [x]
  (let [input-data (:data x)]
    (node (t/silu input-data)
          [x]
          (fn [self]
            (when-let [g @(:grad self)]
              (let [xs (arr/->vec input-data)
                    gs (arr/->vec g)
                    dx (mapv (fn [value upstream]
                               (let [sigmoid (/ 1.0 (+ 1.0 (Math/exp (- (double value)))))
                                     derivative (* sigmoid
                                                   (+ 1.0 (* value (- 1.0 sigmoid))))]
                                 (* upstream derivative)))
                             xs gs)]
                (accumulate! x (arr/from-vec (:backend input-data) dx
                                             (:shape input-data)))))))))

(defn group-norm-nchw*
  "Differentiable PyTorch-style GroupNorm. Optional affine `weight`/`bias`
  are Values with shape `[C]`; gradients are accumulated for input and every
  supplied affine parameter."
  ([x num-groups] (group-norm-nchw* x num-groups nil nil 1.0e-5))
  ([x num-groups weight bias eps]
   (let [input-data (:data x)
         weight-data (when weight (:data weight))
         bias-data (when bias (:data bias))
         output (t/group-norm-nchw input-data num-groups weight-data bias-data eps)
         [N C H W] (mapv long (:shape input-data))
         groups (long num-groups)
         channels-per-group (quot C groups)
         group-size (* channels-per-group H W)]
     (node output
           (cond-> [x] weight (conj weight) bias (conj bias))
           (fn [self]
             (when-let [g @(:grad self)]
               (let [xs (double-array (arr/->vec input-data))
                     gs (double-array (arr/->vec g))
                     ws (when weight (double-array (arr/->vec weight-data)))
                     dx (double-array (* N C H W))
                     dw (when weight (double-array C))
                     dbias (when bias (double-array C))]
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
                           inv-std (/ 1.0 (Math/sqrt (+ variance eps)))
                           sum-dy (loop [i 0 sum 0.0]
                                    (if (< i group-size)
                                      (let [channel (+ (* group channels-per-group)
                                                       (quot i (* H W)))
                                            gamma (if ws (aget ws channel) 1.0)]
                                        (recur (inc i) (+ sum (* (aget gs (+ base i)) gamma))))
                                      sum))
                           sum-dy-xhat
                           (loop [i 0 sum 0.0]
                             (if (< i group-size)
                               (let [channel (+ (* group channels-per-group)
                                                (quot i (* H W)))
                                     gamma (if ws (aget ws channel) 1.0)
                                     xhat (* (- (aget xs (+ base i)) mean) inv-std)]
                                 (recur (inc i)
                                        (+ sum (* (aget gs (+ base i)) gamma xhat))))
                               sum))]
                       (dotimes [i group-size]
                         (let [index (+ base i)
                               channel (+ (* group channels-per-group)
                                          (quot i (* H W)))
                               upstream (aget gs index)
                               gamma (if ws (aget ws channel) 1.0)
                               xhat (* (- (aget xs index) mean) inv-std)
                               dx-value (* (/ inv-std group-size)
                                           (- (* group-size upstream gamma)
                                              sum-dy
                                              (* xhat sum-dy-xhat)))]
                           (aset dx index dx-value)
                           (when dw
                             (aset dw channel (+ (aget dw channel) (* upstream xhat))))
                           (when dbias
                             (aset dbias channel (+ (aget dbias channel) upstream))))))))
                 (accumulate! x (arr/from-vec (:backend input-data) (vec dx)
                                              (:shape input-data)))
                 (when weight
                   (accumulate! weight (arr/from-vec (:backend weight-data) (vec dw)
                                                     (:shape weight-data))))
                 (when bias
                   (accumulate! bias (arr/from-vec (:backend bias-data) (vec dbias)
                                                   (:shape bias-data)))))))))))

(defn cat*
  "Differentiable `num.tensor/cat`. Backward slices the upstream gradient
  along the concatenation axis and accumulates one contiguous gradient per
  input Value."
  [values axis]
  (let [values (vec values)
        data (mapv :data values)
        output (t/cat data axis)
        rank (count (:shape output))
        axis (long (if (neg? axis) (+ rank axis) axis))
        first-shape (:shape (first data))
        inner (long (arr/nelems (subvec first-shape (inc axis))))
        outer (long (arr/nelems (subvec first-shape 0 axis)))
        output-axis (long (nth (:shape output) axis))
        axis-sizes (mapv #(long (nth (:shape %) axis)) data)]
    (node output values
          (fn [self]
            (when-let [g @(:grad self)]
              (let [gs (double-array (arr/->vec g))]
                (loop [value-index 0 axis-offset 0]
                  (when (< value-index (count values))
                    (let [value (nth values value-index)
                          shape (:shape (:data value))
                          axis-size (nth axis-sizes value-index)
                          block (* axis-size inner)
                          sliced (double-array (arr/nelems shape))]
                      (dotimes [outer-index outer]
                        (let [source-base (+ (* outer-index output-axis inner)
                                             (* axis-offset inner))
                              destination-base (* outer-index block)]
                          (dotimes [i block]
                            (aset sliced (+ destination-base i)
                                  (aget gs (+ source-base i))))))
                      (accumulate! value
                                   (arr/from-vec (:backend (:data value))
                                                 (vec sliced) shape))
                      (recur (inc value-index) (+ axis-offset axis-size)))))))))))

(defn upsample-nearest2d*
  "Differentiable nearest-neighbor NCHW upsampling. Backward sums every
  repeated output cell into its source input cell."
  [x scale-factor]
  (let [input-data (:data x)
        [N C H W] (mapv long (:shape input-data))
        [scale-h scale-w] (pair-option scale-factor)
        output (t/upsample-nearest2d input-data scale-factor)
        [_ _ oh ow] (mapv long (:shape output))]
    (node output [x]
          (fn [self]
            (when-let [g @(:grad self)]
              (let [gs (double-array (arr/->vec g))
                    dx (double-array (* N C H W))]
                (dotimes [n N]
                  (dotimes [c C]
                    (dotimes [oi oh]
                      (dotimes [oj ow]
                        (let [input-index (+ (* n C H W) (* c H W)
                                             (* (quot oi scale-h) W)
                                             (quot oj scale-w))
                              output-index (+ (* n C oh ow) (* c oh ow)
                                              (* oi ow) oj)]
                          (aset dx input-index
                                (+ (aget dx input-index) (aget gs output-index))))))))
                (accumulate! x (arr/from-vec (:backend input-data) (vec dx)
                                             (:shape input-data)))))))))
