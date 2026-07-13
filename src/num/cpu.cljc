(ns num.cpu
  "The CPU REFERENCE backend — pure `.cljc` over flat `double-array`s, zero deps.

  It is deliberately simple and obviously-correct: scalar triple loops, no SIMD,
  no threads. Its job is not speed but to be the ORACLE — the contract test
  (`num.contract`) runs the identical suite against this and against every GPU
  backend, asserting `GpuBackend ≡ CpuBackend` to ~f32 tolerance, the same way the
  actors guarantee `MemStore ≡ DatomicStore`. A handle here is just a
  `double-array`; CSR arrays travel as host `int-array`/`double-array`."
  (:require [num.dtype :as dtype]
            [num.protocol :as p]))

(defn- ^double aget* [^doubles a ^long i] (aget a i))

(defn- typed-values [handle]
  (let [data (:data handle)
        decode (case (:dtype handle)
                 :f16 dtype/f16-bits->f32
                 :bf16 dtype/bf16-bits->f32)]
    (mapv #(decode (aget data %)) (range (alength data)))))

(defn- typed-handle [dtype* values]
  (let [encode (case dtype*
                 :f16 dtype/f32->f16-bits
                 :bf16 dtype/f32->bf16-bits)
        encoded (mapv encode values)]
    {:dtype dtype*
     :data #?(:clj (short-array encoded)
              :cljs (js/Int16Array. (into-array encoded)))}))

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
              :silu (fn [v] (/ v (+ 1.0 (Math/exp (- v)))))
              :sigmoid (fn [v] (/ 1.0 (+ 1.0 (Math/exp (- v)))))
              :tanh #(Math/tanh %)
              :sigmoid-gradient (fn [y] (* y (- 1.0 y)))
              :tanh-gradient (fn [y] (- 1.0 (* y y))))]
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
      y))

  p/IDTypeStorage
  (-alloc-dtype [_ n dtype*]
    {:dtype dtype*
     :data #?(:clj (short-array (long n))
              :cljs (js/Int16Array. n))})
  (-copy-from-host-dtype [_ xs dtype*]
    (let [encoded (mapv (case dtype*
                          :f16 dtype/f32->f16-bits
                          :bf16 dtype/f32->bf16-bits)
                        xs)]
      {:dtype dtype*
       :data #?(:clj (short-array encoded)
                :cljs (js/Int16Array. (into-array encoded)))}))
  (-copy-to-host-dtype [_ h n dtype*]
    (when-not (= dtype* (:dtype h))
      (throw (ex-info "typed handle dtype mismatch"
                      {:expected dtype* :actual (:dtype h)})))
    (let [data (:data h)
          decode (case dtype*
                   :f16 dtype/f16-bits->f32
                   :bf16 dtype/bf16-bits->f32)]
      (mapv #(decode (aget data %)) (range (long n)))))

  p/IDTypeOps
  (-ewise-dtype [_ op xh yh n dtype*]
    (let [xs (typed-values xh) ys (typed-values yh)
          f (case op :add + :sub - :mul * :div /)]
      (typed-handle dtype* (mapv f (take n xs) (take n ys)))))
  (-ewise1-dtype [_ op xh n dtype*]
    (let [f (case op
              :exp #(Math/exp %)
              :relu #(max % 0.0)
              :neg -
              :silu (fn [v] (/ v (+ 1.0 (Math/exp (- v)))))
              :sigmoid (fn [v] (/ 1.0 (+ 1.0 (Math/exp (- v)))))
              :tanh #(Math/tanh %)
              :sigmoid-gradient (fn [y] (* y (- 1.0 y)))
              :tanh-gradient (fn [y] (- 1.0 (* y y))))]
      (typed-handle dtype* (mapv f (take n (typed-values xh))))))
  (-gemm-dtype [_ Ah m k Bh n dtype*]
    (let [A (typed-values Ah) B (typed-values Bh)]
      (typed-handle
       dtype*
       (vec
        (for [i (range m) j (range n)]
          (reduce + 0.0
                  (map (fn [l]
                         (* (nth A (+ (* i k) l))
                            (nth B (+ (* l n) j))))
                       (range k)))))))))

(defn cpu-backend
  "Construct the pure-Clojure CPU reference backend."
  []
  (->CpuBackend))
