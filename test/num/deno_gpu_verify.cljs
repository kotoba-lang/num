(ns num.deno-gpu-verify
  "S2 (ADR-2607051400 §Phase 2): cross-checks a REAL `num.deno-gpu/WgslBackendAsync`
  — running on real GPU hardware via Deno's native `navigator.gpu` (wgpu→Metal) —
  against `num.cpu`'s reference oracle, dispatching through `num.core`/`num.array`
  (i.e. through `num.protocol/IBackend`, the same seam any real caller uses), not a
  hand-rolled JS harness. Uses the exact fixtures `num.contract/verify` uses and its
  same `approx?`/`approx-vec?` tolerance, so 'GPU ≡ CPU-oracle' means the same thing
  it means everywhere else in this repo.

  This is NOT `num.contract/verify` reused directly: that suite's `check` callback
  assumes synchronous results, but `WgslBackendAsync`'s host-value-returning ops
  (`dot`/`nrm2`/`sum`/`amax`/`amin`, and `arr/->vec`) return JS Promises (see
  num.deno-gpu's docstring for why). So the CPU-oracle values are computed EAGERLY
  up front (mirroring num.contract/verify's exact fixture order/mutation sequence,
  so in-place ops like axpy!/scal! see the same before/after values num.contract
  asserts) and only the GPU-side comparisons are deferred behind `.then`.

  Run under Deno (the only host with GPU access):
    clojure -M:deno-verify
    deno run --allow-all target/deno-gpu-verify.cjs"
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.sparse :as sp]
            [num.cpu :as cpu]
            [num.contract :as contract]
            [num.deno-gpu :as dg]))

(defn- ->p [x] (if (instance? js/Promise x) x (js/Promise.resolve x)))

(defn- record! [pass fail label ok?]
  (println (str (if ok? "✓" "✗") " " label))
  (if ok? (swap! pass inc) (swap! fail inc)))

(defn -main [& _]
  (let [cpu-b (cpu/cpu-backend)
        ;; --- CPU ORACLE: computed eagerly, same literals + same order as
        ;; num.contract/verify, so in-place mutation (axpy!/scal!) matches it.
        x (arr/from-vec cpu-b [1 2 3 4] [4])       y (arr/from-vec cpu-b [10 20 30 40] [4])
        exp-dot  (nm/dot x y)
        exp-nrm2 (nm/nrm2 (arr/from-vec cpu-b [3 4] [2]))
        exp-axpy (arr/->vec (nm/axpy! 2.0 x y))
        exp-scal (arr/->vec (nm/scal! 2.0 (arr/from-vec cpu-b [1 2 3 4] [4])))
        a (arr/from-vec cpu-b [1 2 3 4] [4])       b (arr/from-vec cpu-b [4 3 2 1] [4])
        exp-add (arr/->vec (nm/add a b))
        exp-sub (arr/->vec (nm/sub a b))
        exp-mul (arr/->vec (nm/mul a b))
        exp-div (arr/->vec (nm/div a b))
        exp-sum (nm/sum a) exp-amax (nm/amax a) exp-amin (nm/amin a)
        c (arr/from-vec cpu-b [0 1 -2 3] [4])
        exp-exp (arr/->vec (nm/exp c))
        exp-relu (arr/->vec (nm/relu c))
        exp-neg (arr/->vec (nm/neg c))
        A (arr/from-vec cpu-b [1 2 3 4] [2 2])
        xv (arr/from-vec cpu-b [1 1] [2])
        B (arr/from-vec cpu-b [5 6 7 8] [2 2])
        exp-matvec (arr/->vec (nm/matvec A xv))
        exp-matmul (arr/->vec (nm/matmul A B))
        csr (sp/dense->csr 2 3 [1 0 2 0 3 0])
        xs (arr/from-vec cpu-b [1 1 1] [3])
        exp-spmv (arr/->vec (nm/spmv cpu-b csr xs))]
    (-> (dg/request-device)
        (.then
         (fn [r]
           (println "GPU:" (dg/adapter-description r) "(Deno navigator.gpu → wgpu → Metal)\n")
           (let [gpu (dg/backend r)
                 pass (atom 0) fail (atom 0)
                 ;; --- GPU side, dispatched through the exact same num.core/
                 ;; num.array API, same fixture order, so submission order (which
                 ;; matters for the in-place axpy!/scal! ops on a single command
                 ;; queue) matches the CPU-oracle's mutation order above.
                 xg (arr/from-vec gpu [1 2 3 4] [4])   yg (arr/from-vec gpu [10 20 30 40] [4])
                 ag (arr/from-vec gpu [1 2 3 4] [4])   bg (arr/from-vec gpu [4 3 2 1] [4])
                 Ag (arr/from-vec gpu [1 2 3 4] [2 2])
                 xvg (arr/from-vec gpu [1 1] [2])
                 Bg (arr/from-vec gpu [5 6 7 8] [2 2])
                 xsg (arr/from-vec gpu [1 1 1] [3])
                 cg (arr/from-vec gpu [0 1 -2 3] [4])
                 checks
                 [["dot"    (->p (nm/dot xg yg))                              (fn [g] (contract/approx? g exp-dot))]
                  ["nrm2"   (->p (nm/nrm2 (arr/from-vec gpu [3 4] [2])))      (fn [g] (contract/approx? g exp-nrm2))]
                  ["axpy!"  (->p (arr/->vec (nm/axpy! 2.0 xg yg)))            (fn [g] (contract/approx-vec? g exp-axpy))]
                  ["scal!"  (->p (arr/->vec (nm/scal! 2.0 (arr/from-vec gpu [1 2 3 4] [4]))))
                                                                              (fn [g] (contract/approx-vec? g exp-scal))]
                  ["add"    (->p (arr/->vec (nm/add ag bg)))                  (fn [g] (contract/approx-vec? g exp-add))]
                  ["sub"    (->p (arr/->vec (nm/sub ag bg)))                  (fn [g] (contract/approx-vec? g exp-sub))]
                  ["mul"    (->p (arr/->vec (nm/mul ag bg)))                  (fn [g] (contract/approx-vec? g exp-mul))]
                  ["div"    (->p (arr/->vec (nm/div ag bg)))                  (fn [g] (contract/approx-vec? g exp-div))]
                  ["sum"    (->p (nm/sum ag))                                 (fn [g] (contract/approx? g exp-sum))]
                  ["amax"   (->p (nm/amax ag))                                (fn [g] (contract/approx? g exp-amax))]
                  ["amin"   (->p (nm/amin ag))                                (fn [g] (contract/approx? g exp-amin))]
                  ["matvec" (->p (arr/->vec (nm/matvec Ag xvg)))              (fn [g] (contract/approx-vec? g exp-matvec))]
                  ["matmul" (->p (arr/->vec (nm/matmul Ag Bg)))               (fn [g] (contract/approx-vec? g exp-matmul))]
                  ["spmv"   (->p (arr/->vec (nm/spmv gpu csr xsg)))           (fn [g] (contract/approx-vec? g exp-spmv))]
                  ["exp"    (->p (arr/->vec (nm/exp cg)))                     (fn [g] (contract/approx-vec? g exp-exp))]
                  ["relu"   (->p (arr/->vec (nm/relu cg)))                    (fn [g] (contract/approx-vec? g exp-relu))]
                  ["neg"    (->p (arr/->vec (nm/neg cg)))                     (fn [g] (contract/approx-vec? g exp-neg))]]]
             (-> (js/Promise.all
                  (into-array
                   (map (fn [[label prom okfn]]
                          (.then prom (fn [g] (record! pass fail label (okfn g)))))
                        checks)))
                 (.then (fn [_]
                          (println (str "\nDeno WgslBackendAsync ≡ CPU oracle: " @pass " passed, " @fail " failed"))
                          (js/Deno.exit (if (pos? @fail) 1 0))))))))
        (.catch (fn [e]
                  (println "ERROR:" (or (.-stack e) (.-message e) (str e)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
