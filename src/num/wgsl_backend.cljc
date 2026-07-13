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
  WebGPU path is async — `num.deno-gpu/WgslBackendAsync` (ADR-2607051400 §Phase 2)
  reuses `ceil-div`/`uni`/`get-pipeline` below (public for that reuse) and mirrors
  this dispatch logic exactly, except the host-value-returning ops
  (`-copy-to-host`/`-reduce`/`-nrm2`) return a JS Promise instead of an immediate
  value, because a Deno/browser `IGpuDevice`'s `-read-buffer` cannot block."
  (:require [num.protocol :as p]
            [num.wgsl :as w]
            [num.cpu :as cpu]))

(defn ceil-div
  "⌈a/b⌉ for workgroup-count sizing. Public: shared by num.deno-gpu's async backend."
  [a b] (quot (+ (long a) (long b) -1) (long b)))

(defn u32-tag
  "Tag `xs` as u32-typed payload (index/dims/op-selector data) via metadata, so an
  IGpuDevice's `-write-buffer` can pick the correct byte pattern — a plain Clojure
  number carries no int-vs-float distinction under ClojureScript (unlike the JVM,
  where Long vs Double could disambiguate this; num.deno-gpu's IGpuDevice runs as
  cljs, so it cannot use that trick). Untagged `xs` defaults to f32 (every payload
  data buffer — vector/matrix values, CSR `vals` — is f32; only CSR `row-ptr`/
  `col-idx` and the uniform dims/op-selector below are u32). Public: shared by
  num.deno-gpu's async backend."
  [xs]
  (with-meta (vec xs) {:num.wgsl/dtype :u32}))

(defn uni
  "Create + fill a uniform buffer from host seq `xs` (device pads to alignment).
  `xs` defaults to f32; pass it through `u32-tag` first for a u32 uniform (dims,
  op-selector). Public: shared by num.deno-gpu's async backend."
  [dev xs]
  (let [b (w/-create-buffer dev (count xs) :uniform)] (w/-write-buffer dev b xs) b))

(defn get-pipeline
  "Lazily compile + cache the pipeline for op keyword `op`.
  Public: shared by num.deno-gpu's async backend."
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
    ;; Force `double` here exactly like num.cpu's -copy-from-host does: callers
    ;; routinely pass literal integers (e.g. `(arr/from-vec b [1 2 3 4] [4])`
    ;; in num.contract's own fixtures) that MUST land as f32 bit patterns on the
    ;; GPU (every num.wgsl shader declares its data buffers `array<f32>`), not
    ;; be reinterpreted as u32 by an IGpuDevice that infers dtype from Clojure's
    ;; runtime number type (see num.deno-gpu). Pre-existing gap fixed in
    ;; ADR-2607051400 §Phase 2 — never caught before because no live device
    ;; exercised this path.
    (let [b (w/-create-buffer dev (count xs) :storage)] (w/-write-buffer dev b (map double xs)) b))
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
                   [xh yh z (uni dev (u32-tag [({:add 0 :sub 1 :mul 2 :div 3} op)]))]
                   [(ceil-div n 64) 1 1])
      z))

  (-ewise1 [_ op xh n]
    (let [z (w/-create-buffer dev n :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :ewise1)
                   [xh z (uni dev (u32-tag [({:exp 0 :relu 1 :neg 2 :silu 3} op)]))]
                   [(ceil-div n 64) 1 1])
      z))

  (-reduce [_ op xh n]
    (let [nwg (ceil-div n 256)
          parts (w/-create-buffer dev nwg :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :reduce)
                   [xh parts (uni dev (u32-tag [({:sum 0 :max 1 :min 2} op)]))] [nwg 1 1])
      (reduce (case op :sum + :max max :min min) (w/-read-buffer dev parts nwg))))

  (-dot [this xh yh n] (p/-reduce this :sum (p/-ewise this :mul xh yh n) n))
  (-nrm2 [this xh n] (Math/sqrt (p/-dot this xh xh n)))

  (-gemv [this alpha Ah m n xh beta yh]
    (w/-dispatch dev (get-pipeline dev pipes :gemv) [Ah xh yh (uni dev (u32-tag [m n]))]
                 [(ceil-div m 64) 1 1])
    (when (or (not= 1.0 alpha) (not= 0.0 beta)) (p/-scal this alpha yh m))
    yh)

  (-gemm [this alpha Ah m k Bh n beta Ch]
    (w/-dispatch dev (get-pipeline dev pipes :gemm) [Ah Bh Ch (uni dev (u32-tag [m k n 0]))]
                 [(ceil-div n 16) (ceil-div m 16) 1])
    (when (not= 1.0 alpha) (p/-scal this alpha Ch (* m n)))
    Ch)

  (-spmv [_ csr xh]
    (let [m (:n-rows csr)
          rp (w/-create-buffer dev (inc m) :storage)
          ci (w/-create-buffer dev (:nnz csr) :storage)
          v  (w/-create-buffer dev (:nnz csr) :storage)
          y  (w/-create-buffer dev m :storage)]
      (w/-write-buffer dev rp (u32-tag (seq (:row-ptr csr))))
      (w/-write-buffer dev ci (u32-tag (seq (:col-idx csr))))
      (w/-write-buffer dev v (seq (:vals csr)))
      (w/-dispatch dev (get-pipeline dev pipes :spmv) [rp ci v xh y] [(ceil-div m 64) 1 1])
      y))

  p/ITensorBackend
  (-conv2d-nchw [_ input-h weight-h bias-h
                 {:keys [n cin h width cout cin-group kh kw oh ow sh sw ph pw dh dw groups]}]
    (let [total (* n cout oh ow)
          output (w/-create-buffer dev total :storage)
          bias (or bias-h (w/-create-buffer dev cout :storage))
          params [n cin h width cout cin-group kh kw oh ow sh sw ph pw dh dw
                  groups 0 0 0]]
      (w/-dispatch dev (get-pipeline dev pipes :conv2d-nchw)
                   [input-h weight-h bias output (uni dev (u32-tag params))]
                   [(ceil-div total 64) 1 1])
      output))
  (-group-norm-nchw [_ input-h weight-h bias-h
                     {:keys [n c h width groups channels-group group-size eps]}]
    (let [total (* n c h width)
          output (w/-create-buffer dev total :storage)
          weight (or weight-h
                     (let [buffer (w/-create-buffer dev c :storage)]
                       (w/-write-buffer dev buffer (repeat c 1.0)) buffer))
          bias (or bias-h (w/-create-buffer dev c :storage))
          dims [n c h width groups channels-group group-size (* h width)]]
      (w/-dispatch dev (get-pipeline dev pipes :group-norm-nchw)
                   [input-h weight bias output (uni dev (u32-tag dims))
                    (uni dev [(double eps)])]
                   [(* n groups) 1 1])
      output))
  (-upsample-nearest2d [_ input-h {:keys [n c h width oh ow scale-h scale-w]}]
    (let [total (* n c oh ow)
          output (w/-create-buffer dev total :storage)
          dims [n c h width oh ow scale-h scale-w]]
      (w/-dispatch dev (get-pipeline dev pipes :upsample-nearest2d)
                   [input-h output (uni dev (u32-tag dims))]
                   [(ceil-div total 64) 1 1])
      output))
  (-cat [_ input-handles {:keys [total-output output-block inputs]}]
    (let [output (w/-create-buffer dev total-output :storage)]
      (doseq [[input-h {:keys [total block axis-offset]}]
              (map vector input-handles inputs)]
        (w/-dispatch dev (get-pipeline dev pipes :cat-copy)
                     [input-h output (uni dev (u32-tag [total block output-block axis-offset]))]
                     [(ceil-div total 64) 1 1]))
      output))
  (-add-last-axis-bias [_ input-h bias-h {:keys [total width]}]
    (let [output (w/-create-buffer dev total :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :add-last-axis-bias)
                   [input-h bias-h output (uni dev (u32-tag [total width 0 0]))]
                   [(ceil-div total 64) 1 1])
      output))
  (-multi-head-attention [_ query-h key-h value-h
                          {:keys [seq-q seq-k d-model heads head-dim total]}]
    (let [output (w/-create-buffer dev total :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :multi-head-attention)
                   [query-h key-h value-h output
                    (uni dev (u32-tag [seq-q seq-k d-model heads
                                       head-dim total 0 0]))]
                   [(ceil-div total 64) 1 1])
      output))
  (-multi-head-attention-backward [_ query-h key-h value-h grad-output-h
                                   {:keys [seq-q seq-k d-model heads head-dim]}]
    (let [total-q (* seq-q d-model)
          total-k (* seq-k d-model)
          total (max total-q total-k)
          grad-query (w/-create-buffer dev total-q :storage)
          grad-key (w/-create-buffer dev total-k :storage)
          grad-value (w/-create-buffer dev total-k :storage)]
      (w/-dispatch dev (get-pipeline dev pipes :multi-head-attention-backward)
                   [query-h key-h value-h grad-output-h
                    grad-query grad-key grad-value
                    (uni dev (u32-tag [seq-q seq-k d-model heads head-dim
                                       total-q total-k total]))]
                   [(ceil-div total 64) 1 1])
      {:query grad-query :key grad-key :value grad-value})))

(defn wgsl-backend
  "Construct a WgslBackend over an injected `IGpuDevice` (native, blocking
  readback). Pipelines compile lazily on first use."
  [device]
  (->WgslBackend device (atom {}) (cpu/cpu-backend)))
