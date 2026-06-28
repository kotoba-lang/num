(ns num.wgsl-backend
  "WgslBackend — a host-side `IBackend` that runs num-clj on the GPU by dispatching
  the `num.wgsl` compute shaders through an injected `IGpuDevice`. It is the
  Clojure dispatch logic; the shaders themselves are verified on Apple M4 Metal
  (`verify/metal_contract.js`), and a live device makes this deftype reproduce that
  contract on any wgpu target.

  Sync vs async: this backend is SYNCHRONOUS (it returns host values from
  `-copy-to-host`). WebGPU buffer readback is fundamentally async, so a synchronous
  WgslBackend needs a device whose `-read-buffer` BLOCKS — i.e. a native wgpu
  binding (JVM Panama/FFM → wgpu-native) or a vendor backend. The browser/Deno
  WebGPU path is async and is exercised by the JS verify harness instead."
  (:require [num.protocol :as p]
            [num.wgsl :as w]
            [num.cpu :as cpu]))

(defn- ceil-div [a b] (quot (+ (long a) (long b) -1) (long b)))

(defn- uni
  "Create + fill a uniform buffer from host seq `xs` (device pads to alignment)."
  [dev xs]
  (let [b (w/-create-buffer dev (count xs) :uniform)] (w/-write-buffer dev b xs) b))

(defn- get-pipeline
  "Lazily compile + cache the pipeline for op keyword `op`."
  [dev pipes op]
  (or (get @pipes op)
      (let [pl (w/-compile dev (get w/shaders op) "main")]
        (swap! pipes assoc op pl) pl)))

(deftype WgslBackend [dev pipes fallback]
  p/IBackend
  (-backend-name [_] :wgsl)
  (-alloc [_ n] (w/-create-buffer dev n :storage))
  (-free [_ _] nil)
  (-copy-from-host [_ xs]
    (let [b (w/-create-buffer dev (count xs) :storage)] (w/-write-buffer dev b xs) b))
  (-copy-to-host [_ h n] (w/-read-buffer dev h n))      ; BLOCKING readback (native device)

  (-axpy [_ alpha xh yh n]
    (w/-dispatch dev (get-pipeline dev pipes :axpy) [xh yh (uni dev [(double alpha)])]
                 [(ceil-div n 64) 1 1])
    yh)

  (-scal [_ alpha xh n]
    (w/-dispatch dev (get-pipeline dev pipes :scal) [xh (uni dev [(double alpha)])]
                 [(ceil-div n 64) 1 1])
    xh)

  (-ewise [_ op xh yh n]
    (let [z (w/-create-buffer dev n :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :ewise)
                   [xh yh z (uni dev [({:add 0 :sub 1 :mul 2 :div 3} op)])]
                   [(ceil-div n 64) 1 1])
      z))

  (-reduce [_ op xh n]
    (let [nwg (ceil-div n 256)
          parts (w/-create-buffer dev nwg :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :reduce)
                   [xh parts (uni dev [({:sum 0 :max 1 :min 2} op)])] [nwg 1 1])
      (reduce (case op :sum + :max max :min min) (w/-read-buffer dev parts nwg))))

  (-dot [this xh yh n] (p/-reduce this :sum (p/-ewise this :mul xh yh n) n))
  (-nrm2 [this xh n] (Math/sqrt (p/-dot this xh xh n)))

  (-gemv [this alpha Ah m n xh beta yh]
    (w/-dispatch dev (get-pipeline dev pipes :gemv) [Ah xh yh (uni dev [m n])]
                 [(ceil-div m 64) 1 1])
    (when (or (not= 1.0 alpha) (not= 0.0 beta)) (p/-scal this alpha yh m))
    yh)

  (-gemm [this alpha Ah m k Bh n beta Ch]
    (w/-dispatch dev (get-pipeline dev pipes :gemm) [Ah Bh Ch (uni dev [m k n 0])]
                 [(ceil-div n 16) (ceil-div m 16) 1])
    (when (not= 1.0 alpha) (p/-scal this alpha Ch (* m n)))
    Ch)

  (-spmv [_ csr xh]
    (let [m (:n-rows csr)
          rp (w/-create-buffer dev (inc m) :storage)
          ci (w/-create-buffer dev (:nnz csr) :storage)
          v  (w/-create-buffer dev (:nnz csr) :storage)
          y  (w/-create-buffer dev m :storage)]
      (w/-write-buffer dev rp (seq (:row-ptr csr)))
      (w/-write-buffer dev ci (seq (:col-idx csr)))
      (w/-write-buffer dev v (seq (:vals csr)))
      (w/-dispatch dev (get-pipeline dev pipes :spmv) [rp ci v xh y] [(ceil-div m 64) 1 1])
      y)))

(defn wgsl-backend
  "Construct a WgslBackend over an injected `IGpuDevice` (native, blocking
  readback). Pipelines compile lazily on first use."
  [device]
  (->WgslBackend device (atom {}) (cpu/cpu-backend)))
