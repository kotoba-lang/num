(ns num.cpu
  "The CPU REFERENCE backend — pure `.cljc` over flat `double-array`s, zero deps.

  It is deliberately simple and obviously-correct: scalar triple loops, no SIMD,
  no threads. Its job is not speed but to be the ORACLE — the contract test
  (`num.contract`) runs the identical suite against this and against every GPU
  backend, asserting `GpuBackend ≡ CpuBackend` to ~f32 tolerance, the same way the
  actors guarantee `MemStore ≡ DatomicStore`. A handle here is just a
  `double-array`; CSR arrays travel as host `int-array`/`double-array`."
  (:require [num.protocol :as p]))

(defn- ^double aget* [^doubles a ^long i] (aget a i))

(deftype CpuBackend []
  p/IBackend
  (-backend-name [_] :cpu)
  (-alloc [_ n] (double-array (long n)))
  (-free [_ _] nil)
  (-copy-to-host [_ h n] (let [^doubles a h n (long n)]
                           (vec (for [i (range n)] (aget a i)))))
  (-copy-from-host [_ xs] (double-array (map double xs)))

  (-axpy [_ alpha xh yh n]
    (let [^doubles x xh ^doubles y yh a (double alpha) n (long n)]
      (dotimes [i n] (aset y i (+ (aget y i) (* a (aget x i)))))
      y))

  (-scal [_ alpha xh n]
    (let [^doubles x xh a (double alpha) n (long n)]
      (dotimes [i n] (aset x i (* a (aget x i))))
      x))

  (-dot [_ xh yh n]
    (let [^doubles x xh ^doubles y yh n (long n)]
      (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (aget x i) (aget y i)))) s))))

  (-nrm2 [b xh n] (Math/sqrt (p/-dot b xh xh n)))

  (-ewise [_ op xh yh n]
    (let [^doubles x xh ^doubles y yh n (long n) z (double-array n)
          f (case op :add + :sub - :mul * :div /)]
      (dotimes [i n] (aset z i (double (f (aget x i) (aget y i)))))
      z))

  (-ewise1 [_ op xh n]
    (let [^doubles x xh n (long n) z (double-array n)
          f (case op
              :exp #(Math/exp %)
              :relu #(max % 0.0)
              :neg -
              :silu (fn [v] (/ v (+ 1.0 (Math/exp (- v))))))]
      (dotimes [i n] (aset z i (double (f (aget x i)))))
      z))

  (-reduce [_ op xh n]
    (let [^doubles x xh n (long n)]
      (case op
        :sum (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (aget x i))) s))
        :max (loop [i 1 m (aget x 0)] (if (< i n) (recur (inc i) (max m (aget x i))) m))
        :min (loop [i 1 m (aget x 0)] (if (< i n) (recur (inc i) (min m (aget x i))) m)))))

  (-gemv [_ alpha Ah m n xh beta yh]
    (let [^doubles A Ah ^doubles x xh ^doubles y yh
          a (double alpha) bt (double beta) m (long m) n (long n)]
      (dotimes [i m]
        (let [row (* i n)
              s (loop [j 0 s 0.0] (if (< j n) (recur (inc j) (+ s (* (aget A (+ row j)) (aget x j)))) s))]
          (aset y i (+ (* a s) (* bt (aget y i))))))
      y))

  (-gemm [_ alpha Ah m k Bh n beta Ch]
    (let [^doubles A Ah ^doubles B Bh ^doubles C Ch
          a (double alpha) bt (double beta) m (long m) k (long k) n (long n)]
      (dotimes [i m]
        (dotimes [j n]
          (let [s (loop [l 0 s 0.0]
                    (if (< l k)
                      (recur (inc l) (+ s (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))
                      s))]
            (aset C (+ (* i n) j) (+ (* a s) (* bt (aget C (+ (* i n) j))))))))
      C))

  (-spmv [_ csr xh]
    (let [^doubles x xh
          ^ints rp (:row-ptr csr) ^ints ci (:col-idx csr) ^doubles v (:vals csr)
          m (long (:n-rows csr)) y (double-array m)]
      (dotimes [i m]
        (let [a (aget rp i) z (aget rp (inc i))]
          (aset y i (loop [p a s 0.0]
                      (if (< p z) (recur (inc p) (+ s (* (aget v p) (aget x (aget ci p))))) s)))))
      y)))

(defn cpu-backend
  "Construct the pure-Clojure CPU reference backend."
  []
  (->CpuBackend))
