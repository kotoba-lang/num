(ns num.tensor-async
  "Closes the KNOWN GAP `num.tensor`'s own docstring documents (found live
  2026-07-13, ADR-2607131500): that namespace's host round-trips assume
  `num.array/->vec` resolves synchronously, which `num.deno-gpu`'s
  WgslBackendAsync — the only currently-live real-Metal device — does not
  (WebGPU buffer readback is inherently async). Confirmed live: calling
  `num.tensor/softmax` against the Deno GPU backend threw
  `[object Promise] is not ISeqable`.

  This does NOT generalize `num.tensor` to be async-aware (that would mean
  Promise-chaining its ENTIRE N-D broadcast/reduce/batched-matmul machinery
  — materially bigger scope). It closes the gap for EXACTLY the ops
  `comfyui.nodes.toy-diffusion`'s real node needs: 2-D transpose, 2-D last-
  axis softmax, single-channel conv2d, single-head attention — each a
  self-contained async host round-trip mirroring its `num.tensor` sync
  counterpart's exact math (so a discrepancy this uncovers is a genuine
  backend bug, not a reimplementation-drift bug), PLUS reuse of
  `num.core/matmul` UNCHANGED: `-gemm`/`-alloc` are fully synchronous even on
  `WgslBackendAsync` (only `-reduce`/`-dot`/`-nrm2`/`-copy-to-host` are
  async, per that namespace's own docstring) — so plain 2-D matmul already
  worked on the live GPU device with zero modification, verified by the
  14/14 in `num.deno-gpu-verify`.

  CLJS-only (needs `navigator.gpu`/Deno, same as `num.deno-gpu`); the `:clj`
  branch is an informative stub, same convention as that namespace."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.tensor :as t]))

#?(:cljs
   (do
     (defn- ->p [x] (if (instance? js/Promise x) x (js/Promise.resolve x)))

     (defn transpose-2d-async
       "Async twin of `num.tensor/transpose` for exactly the 2-D no-perm
       (full-reverse) case — the only case `attention-async` needs."
       [a]
       (let [[m n] (:shape a)]
         (.then (->p (arr/->vec a))
                (fn [xs]
                  (let [xs (vec xs) out (double-array (* (long m) (long n)))]
                    (dotimes [i m]
                      (dotimes [j n]
                        (aset out (+ (* j m) i) (nth xs (+ (* i n) j)))))
                    (arr/from-vec (:backend a) (vec out) [n m]))))))

     (defn softmax-2d-async
       "Async twin of `num.tensor/softmax` for exactly the 2-D last-axis
       (row-wise) case. Numerically stable (subtract-max), same formula as
       the sync version — computed as one self-contained host round-trip
       rather than composed from broadcasting sub/div (which would each need
       their own async variant); see this ns's own docstring for why that
       narrower scope is the honest tradeoff here."
       [a]
       (let [[batch classes] (:shape a)]
         (.then (->p (arr/->vec a))
                (fn [xs]
                  (let [xs (vec xs) classes (long classes)
                        out (double-array (* (long batch) classes))]
                    (dotimes [b batch]
                      (let [row (subvec xs (* b classes) (* (inc b) classes))
                            mx (reduce max row)
                            ex (mapv #(Math/exp (- % mx)) row)
                            s (reduce + ex)]
                        (dotimes [c classes]
                          (aset out (+ (* b classes) c) (/ (nth ex c) s)))))
                    (arr/from-vec (:backend a) (vec out) (:shape a)))))))

     (defn conv2d-async
       "Async twin of `num.tensor/conv2d` — same im2col-into-matmul shape,
       async patch-extraction read. `nm/matmul` itself needs no async
       variant (see ns docstring)."
       [a kernel]
       (let [[H W] (:shape a) [kh kw] (:shape kernel)
             oh (inc (- (long H) (long kh))) ow (inc (- (long W) (long kw)))]
         (when (or (< oh 1) (< ow 1))
           (throw (ex-info "num.tensor-async/conv2d-async: kernel larger than input"
                           {:input [H W] :kernel [kh kw]})))
         (.then (->p (arr/->vec a))
                (fn [xs]
                  (let [xs (vec xs) W (long W) kh (long kh) kw (long kw)
                        patches (double-array (* oh ow kh kw))]
                    (dotimes [oi oh]
                      (dotimes [oj ow]
                        (dotimes [ki kh]
                          (dotimes [kj kw]
                            (aset patches (+ (* (+ (* oi ow) oj) kh kw) (* ki kw) kj)
                                  (nth xs (+ (* (+ oi ki) W) (+ oj kj))))))))
                    (let [P (arr/from-vec (:backend a) (vec patches) [(* oh ow) (* kh kw)])
                          kflat (t/reshape kernel [(* kh kw) 1])]
                      (t/reshape (nm/matmul P kflat) [oh ow])))))))

     (defn attention-async
       "Async twin of `num.tensor/attention` (single-head scaled dot-product,
       no batching/masking — same scope fence). `Q`/`K`/`V` are `[seq d]`."
       [Q K V]
       (let [d (long (last (:shape Q)))]
         (.then (transpose-2d-async K)
                (fn [Kt]
                  (let [scores (nm/matmul Q Kt)
                        scaled (nm/scal! (/ 1.0 (Math/sqrt d)) scores)]
                    (.then (softmax-2d-async scaled)
                           (fn [weights] (nm/matmul weights V)))))))))

   :clj
   (do
     (defn transpose-2d-async [_a] (throw (ex-info "num.tensor-async/transpose-2d-async requires ClojureScript compiled for a Deno/WebGPU host." {})))
     (defn softmax-2d-async [_a] (throw (ex-info "num.tensor-async/softmax-2d-async requires ClojureScript compiled for a Deno/WebGPU host." {})))
     (defn conv2d-async [_a _kernel] (throw (ex-info "num.tensor-async/conv2d-async requires ClojureScript compiled for a Deno/WebGPU host." {})))
     (defn attention-async [_Q _K _V] (throw (ex-info "num.tensor-async/attention-async requires ClojureScript compiled for a Deno/WebGPU host." {})))))
