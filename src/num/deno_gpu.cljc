(ns num.deno-gpu
  "The Deno WebGPU→Metal `IGpuDevice` HOST (ADR-2607051400 §Phase 2) — promotes
  `verify/metal_contract.js`'s already-proven-on-real-Apple-M4-Metal raw JS harness
  into a REAL, reusable `num.wgsl/IGpuDevice` implementation, so `num.core`'s ops
  (`axpy!`/`scal!`/`add`/`sub`/`mul`/`div`/`sum`/`amax`/`amin`/`matvec`/`matmul`/`spmv`)
  dispatch through the SAME shaders on real GPU hardware, from real Clojure code,
  instead of only being exercised by a standalone verification script.

  CLJS-only (needs `navigator.gpu`, a Deno/browser global — the JVM side is an
  informative stub). Requires `deno run --allow-all` (`navigator.gpu` is native in
  Deno ≥ 2, no `--unstable-webgpu` flag needed — confirmed against the Deno binary
  this ADR pass was built and tested with).

  ## Sync vs. async — the one real design tension, resolved

  WebGPU's device negotiation (`requestAdapter`/`requestDevice`) and buffer readback
  (`mapAsync`) are Promise-based; everything else (`createBuffer`/`writeBuffer`/
  `createComputePipeline`/`dispatchWorkgroups`/`queue.submit`) is SYNCHRONOUS — a
  single GPU queue processes submitted command buffers strictly in submission
  order, so kernel dispatch never needs to wait on anything. Concretely, for every
  `num.protocol/IBackend` method:
    - `-alloc -free -copy-from-host -axpy -scal -ewise -gemv -gemm -spmv` are ALL
      fully synchronous end-to-end on this device (they only ever create/write/
      dispatch, never read back) — dispatching `axpy!`/`scal!`/`add`/`sub`/`mul`/
      `div`/`matvec`/`matmul`/`spmv` through `WgslBackendAsync` behaves EXACTLY
      like the fully-synchronous `num.wgsl-backend/WgslBackend`.
    - `-copy-to-host -reduce -dot -nrm2` are the only ops that ever call
      `-read-buffer`, and `-read-buffer` is unavoidably async on this host (it
      awaits `GPUBuffer.mapAsync`). `WgslBackendAsync` does NOT pretend otherwise:
      these four methods return a JS `Promise` instead of an immediate value — a
      deliberate, documented deviation from `IBackend`'s normal 'reductions/dot
      return host scalars' contract, exactly the tension `num.wgsl-backend`'s own
      docstring already flagged as unresolved. Callers on Deno must `.then` (or
      `await` at the top-level script boundary, which Deno supports) these four;
      see `test/num/deno_gpu_verify.cljs` for the pattern.

  We tried the ClojureScript-native `^:async`/`js-await` special forms first —
  confirmed NOT supported by this repo's plain `clojure -M -m cljs.main` compiler
  pipeline (ClojureScript 1.11.132, no shadow-cljs): they compile to calls on an
  undeclared var, not real `async`/`await` JS. Plain JS Promise `.then` interop
  (the pattern `kotoba-lang/host`'s `kami.backend.browser` already uses for its own
  browser GPU host bring-up, mixed with `cljs.core.async`'s `go` there) is the
  established, working idiom in this org for this exact kind of host-boundary
  async bridging, so `WgslBackendAsync` uses the same interop style, without
  pulling in `core.async` (not needed here — no complex internal control flow, just
  a couple of `.then` chains)."
  (:require [num.protocol :as p]
            [num.wgsl :as w]
            [num.wgsl-backend :as wb]))

#?(:cljs
   (do
     ;; --- IGpuDevice: navigator.gpu, mirroring verify/metal_contract.js exactly ---

     (def ^:private storage-usage
       (bit-or js/GPUBufferUsage.STORAGE js/GPUBufferUsage.COPY_SRC js/GPUBufferUsage.COPY_DST))
     (def ^:private uniform-usage
       (bit-or js/GPUBufferUsage.UNIFORM js/GPUBufferUsage.COPY_DST))
     (def ^:private readback-usage
       (bit-or js/GPUBufferUsage.COPY_DST js/GPUBufferUsage.MAP_READ))

     (defn- usage->flags [usage]
       (case usage
         :uniform uniform-usage
         (:storage :read) storage-usage))

     (deftype DenoGpuDevice [dev]
       w/IGpuDevice
       (-create-buffer [_ n usage]
         (let [nbytes (* 4 (long n))
               floor (if (= usage :uniform) 16 4)]
           (.createBuffer dev #js {:size (max nbytes floor) :usage (usage->flags usage)})))

       (-write-buffer [_ buf xs]
         ;; Dtype comes from `num.wgsl-backend/u32-tag` metadata, NOT from
         ;; inspecting the numbers themselves: ClojureScript numbers have no
         ;; Long-vs-Double distinction (unlike the JVM) — `(integer? 2.0)` is
         ;; TRUE in cljs, so a value-based guess would misclassify every whole-
         ;; number f32 payload (e.g. `[1 2 3 4]`) as u32. Untagged xs defaults to
         ;; f32 (every payload-data buffer is f32; only CSR row-ptr/col-idx and
         ;; the ewise/reduce/gemv/gemm uniform dims/op-selector are u32, and
         ;; num.wgsl-backend explicitly tags exactly those call sites).
         (let [u32? (= :u32 (:num.wgsl/dtype (meta xs)))
               arr (if u32?
                     (js/Uint32Array. (into-array xs))
                     (js/Float32Array. (into-array (map double xs))))]
           (.writeBuffer (.-queue dev) buf 0 arr)))

       (-read-buffer [_ buf n]
         ;; ASYNC — see namespace docstring. Every real -read-buffer call site in
         ;; this repo (-copy-to-host, -reduce's partials) only ever reads f32 data
         ;; (every num.wgsl shader's OUTPUT buffers are `array<f32>`), so this can
         ;; always interpret the readback as Float32Array.
         (let [nbytes (max (* 4 (long n)) 4)
               staging (.createBuffer dev #js {:size nbytes :usage readback-usage})
               encoder (.createCommandEncoder dev)]
           (.copyBufferToBuffer encoder buf 0 staging 0 nbytes)
           (.submit (.-queue dev) #js [(.finish encoder)])
           (-> (.mapAsync staging js/GPUMapMode.READ)
               (.then (fn [_]
                        (let [out (vec (js/Array.from (js/Float32Array. (.slice (.getMappedRange staging) 0))))]
                          (.unmap staging)
                          out))))))

       (-compile [_ wgsl-src entry]
         (let [module (.createShaderModule dev #js {:code wgsl-src})]
           (.createComputePipeline dev #js {:layout "auto"
                                             :compute #js {:module module :entryPoint entry}})))

       (-dispatch [_ pipeline buffers workgroups]
         (let [bind-group (.createBindGroup
                            dev #js {:layout (.getBindGroupLayout pipeline 0)
                                     :entries (into-array
                                               (map-indexed
                                                (fn [i b] #js {:binding i :resource #js {:buffer b}})
                                                buffers))})
               encoder (.createCommandEncoder dev)
               pass (.beginComputePass encoder)
               [wx wy wz] workgroups]
           (.setPipeline pass pipeline)
           (.setBindGroup pass 0 bind-group)
           (.dispatchWorkgroups pass wx wy wz)
           (.end pass)
           (.submit (.-queue dev) #js [(.finish encoder)]))))

     ;; --- device negotiation (the ONLY inherently-async step) ------------------

     (defn request-device
       "Negotiate `navigator.gpu.requestAdapter() → requestDevice()` (exactly
       `metal_contract.js`'s bring-up). Returns a JS Promise<#js{:adapter :device}>."
       []
       (-> (.requestAdapter js/navigator.gpu)
           (.then (fn [adapter]
                    (-> (.requestDevice adapter)
                        (.then (fn [dev] #js {:adapter adapter :device dev})))))))

     (defn adapter-description
       "Best-effort adapter description string for logging (mirrors
       `adapter.info?.description` in the JS verify harness). `r` is the
       `#js{:adapter :device}` `request-device` resolves to."
       [r]
       (some-> r .-adapter .-info .-description))

     ;; --- WgslBackendAsync: num.wgsl-backend/WgslBackend's dispatch logic, over
     ;;     an async device ------------------------------------------------------

     (deftype WgslBackendAsync [dev pipes]
       p/IBackend
       (-backend-name [_] :wgsl-deno)
       (-alloc [_ n] (w/-create-buffer dev n :storage))
       (-free [_ _] nil)
       (-copy-from-host [_ xs]
         (let [b (w/-create-buffer dev (count xs) :storage)] (w/-write-buffer dev b (map double xs)) b))
       (-copy-to-host [_ h n] (w/-read-buffer dev h n))          ; => Promise<vector> — see ns docstring

       (-axpy [_ alpha xh yh n]
         (w/-dispatch dev (wb/get-pipeline dev pipes :axpy) [xh yh (wb/uni dev [(double alpha)])]
                      [(wb/ceil-div n 64) 1 1])
         yh)

       (-scal [_ alpha xh n]
         (w/-dispatch dev (wb/get-pipeline dev pipes :scal) [xh (wb/uni dev [(double alpha)])]
                      [(wb/ceil-div n 64) 1 1])
         xh)

       (-ewise [_ op xh yh n]
         (let [z (w/-create-buffer dev n :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :ewise)
                        [xh yh z (wb/uni dev (wb/u32-tag [({:add 0 :sub 1 :mul 2 :div 3} op)]))]
                        [(wb/ceil-div n 64) 1 1])
           z))

       (-ewise1 [_ op xh n]
         (let [z (w/-create-buffer dev n :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :ewise1)
                        [xh z (wb/uni dev (wb/u32-tag [({:exp 0 :relu 1 :neg 2 :silu 3} op)]))]
                        [(wb/ceil-div n 64) 1 1])
           z))

       (-reduce [_ op xh n]
         (let [nwg (wb/ceil-div n 256)
               parts (w/-create-buffer dev nwg :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :reduce)
                        [xh parts (wb/uni dev (wb/u32-tag [({:sum 0 :max 1 :min 2} op)]))] [nwg 1 1])
           (.then (w/-read-buffer dev parts nwg)                 ; => Promise<number>
                  (fn [xs] (reduce (case op :sum + :max max :min min) xs)))))

       (-dot [this xh yh n] (p/-reduce this :sum (p/-ewise this :mul xh yh n) n)) ; => Promise<number>, automatically
       (-nrm2 [this xh n] (.then (p/-dot this xh xh n) (fn [d] (Math/sqrt d))))   ; => Promise<number>

       (-gemv [this alpha Ah m n xh beta yh]
         (w/-dispatch dev (wb/get-pipeline dev pipes :gemv) [Ah xh yh (wb/uni dev (wb/u32-tag [m n]))]
                      [(wb/ceil-div m 64) 1 1])
         (when (or (not= 1.0 alpha) (not= 0.0 beta)) (p/-scal this alpha yh m))
         yh)

       (-gemm [this alpha Ah m k Bh n beta Ch]
         (w/-dispatch dev (wb/get-pipeline dev pipes :gemm) [Ah Bh Ch (wb/uni dev (wb/u32-tag [m k n 0]))]
                      [(wb/ceil-div n 16) (wb/ceil-div m 16) 1])
         (when (not= 1.0 alpha) (p/-scal this alpha Ch (* m n)))
         Ch)

       (-spmv [_ csr xh]
         (let [m (:n-rows csr)
               rp (w/-create-buffer dev (inc m) :storage)
               ci (w/-create-buffer dev (:nnz csr) :storage)
               v  (w/-create-buffer dev (:nnz csr) :storage)
               y  (w/-create-buffer dev m :storage)]
           (w/-write-buffer dev rp (wb/u32-tag (seq (:row-ptr csr))))
           (w/-write-buffer dev ci (wb/u32-tag (seq (:col-idx csr))))
           (w/-write-buffer dev v (seq (:vals csr)))
           (w/-dispatch dev (wb/get-pipeline dev pipes :spmv) [rp ci v xh y] [(wb/ceil-div m 64) 1 1])
           y)))

     (defn backend
       "Wrap an already-negotiated `r` (`request-device`'s resolved value) as a
       WgslBackendAsync — the num.protocol/IBackend GPU dispatch entry point for
       Deno hosts. Pipelines compile lazily on first use, exactly like
       num.wgsl-backend/wgsl-backend."
       [r]
       (->WgslBackendAsync (->DenoGpuDevice (.-device r)) (atom {})))

     (defn gpu-backend
       "Negotiate a live device AND wrap it as a WgslBackendAsync in one step.
       Mirrors num.cpu/cpu-backend's explicit-construction pattern (num has no
       dynamic `*backend*` var — every call site is passed a backend value, or an
       NDArray carries one in `:backend`); the only difference from `cpu-backend`
       is that THIS constructor is necessarily async (device negotiation is
       Promise-based), so it returns a JS Promise<WgslBackendAsync> rather than an
       immediate value."
       []
       (.then (request-device) backend)))

   :clj
   (do
     (defn request-device [] (throw (ex-info "num.deno-gpu/request-device requires ClojureScript compiled for a Deno/WebGPU host — see README 'Live GPU backend (Deno)'." {})))
     (defn adapter-description [_] (throw (ex-info "num.deno-gpu/adapter-description requires a cljs/Deno host." {})))
     (defn backend [_] (throw (ex-info "num.deno-gpu/backend requires a cljs/Deno host." {})))
     (defn gpu-backend [] (throw (ex-info "num.deno-gpu/gpu-backend requires a cljs/Deno host." {})))))
