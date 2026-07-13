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
            [num.dtype :as dtype]
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
           (.submit (.-queue dev) #js [(.finish encoder)])))

       w/IGpuDeviceLifecycle
       (-destroy-buffer [_ buffer]
         (.destroy buffer))

       w/IGpuDeviceDType
       (-create-buffer-dtype [_ n usage dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "WebGPU typed storage currently supports f16 only"
                           {:dtype dtype*})))
         (.createBuffer dev #js {:size (max (* 4 (quot (+ (long n) 1) 2)) 4)
                                 :usage (usage->flags usage)}))
       (-write-buffer-dtype [_ buf xs dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "unsupported WebGPU dtype" {:dtype dtype*})))
         (let [encoded (js/Uint16Array.
                        (into-array (map #(bit-and (dtype/f32->f16-bits %) 0xffff) xs)))]
           (.writeBuffer (.-queue dev) buf 0 encoded)))
       (-read-buffer-dtype [_ buf n dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "unsupported WebGPU dtype" {:dtype dtype*})))
         (let [nbytes (max (* 4 (quot (+ (long n) 1) 2)) 4)
               staging (.createBuffer dev #js {:size nbytes :usage readback-usage})
               encoder (.createCommandEncoder dev)]
           (.copyBufferToBuffer encoder buf 0 staging 0 nbytes)
           (.submit (.-queue dev) #js [(.finish encoder)])
           (-> (.mapAsync staging js/GPUMapMode.READ)
               (.then (fn [_]
                        (let [raw (js/Uint16Array. (.slice (.getMappedRange staging) 0))
                              out (mapv dtype/f16-bits->f32
                                        (take n (js/Array.from raw)))]
                          (.unmap staging)
                          out)))))))

     ;; --- device negotiation (the ONLY inherently-async step) ------------------

     (defn request-device
       "Negotiate `navigator.gpu.requestAdapter() → requestDevice()` (exactly
       `metal_contract.js`'s bring-up). Returns a JS Promise<#js{:adapter :device}>."
       []
       (-> (.requestAdapter js/navigator.gpu)
           (.then (fn [adapter]
                    (-> (.requestDevice adapter)
                        (.then (fn [dev]
                                 (set! (.-onuncapturederror dev)
                                       (fn [event] (js/console.error (.-error event))))
                                 #js {:adapter adapter :device dev})))))))

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
       (-free [_ h]
         (w/-destroy-buffer dev h)
         nil)
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
                        [xh z (wb/uni
                               dev (wb/u32-tag
                                    [({:exp 0 :relu 1 :neg 2 :silu 3
                                       :sigmoid 4 :tanh 5
                                       :sigmoid-gradient 6 :tanh-gradient 7
                                       :gelu 8 :gelu-gradient 9} op)]))]
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
           y))

       p/IQuantizedOps
       (-quantized-from-host [_ bytes _params]
         (let [words (wb/pack-bytes-u32 bytes)
               buffer (w/-create-buffer dev (count words) :storage)]
           (w/-write-buffer dev buffer words)
           buffer))
       (-quantized-matmul [_ input-h weight-h
                           {:keys [quant-type m k n blocks-per-row]}]
         (when-not (#{:q4-k :q6-k} quant-type)
           (throw (ex-info "unsupported WebGPU quantized matmul"
                           {:quant-type quant-type})))
         (let [output (w/-create-buffer dev (* m n) :storage)]
           (w/-dispatch dev (wb/get-pipeline
                             dev pipes ({:q4-k :q4-k-matmul
                                        :q6-k :q6-k-matmul} quant-type))
                        [input-h weight-h output
                         (wb/uni dev (wb/u32-tag [m k n blocks-per-row]))]
                        [(wb/ceil-div (* m n) 64) 1 1])
           output))

       p/IMutableBufferOps
       (-copy-into! [_ destination source offset n dtype*]
         (case dtype*
           :f32
           (w/-dispatch dev (wb/get-pipeline dev pipes :copy-into)
                        [destination source
                         (wb/uni dev (wb/u32-tag [offset n 0 0]))]
                        [(wb/ceil-div n 64) 1 1])
           :f16
           (do
             (when-not (and (even? offset) (even? n))
               (throw (ex-info "f16 copy-into requires even offset and count"
                               {:offset offset :count n})))
             (w/-dispatch dev (wb/get-pipeline dev pipes :copy-into-f16)
                          [destination source
                           (wb/uni dev (wb/u32-tag
                                        [(quot offset 2) (quot n 2) 0 0]))]
                          [(wb/ceil-div (quot n 2) 64) 1 1]))
           (throw (ex-info "GPU copy-into supports f32/f16" {:dtype dtype*})))
         destination)

       p/IDTypeStorage
       (-alloc-dtype [_ n dtype*]
         (w/-create-buffer-dtype dev n :storage dtype*))
       (-copy-from-host-dtype [_ xs dtype*]
         (let [buffer (w/-create-buffer-dtype dev (count xs) :storage dtype*)]
           (w/-write-buffer-dtype dev buffer xs dtype*)
           buffer))
       (-copy-to-host-dtype [_ h n dtype*]
         (w/-read-buffer-dtype dev h n dtype*))

       p/IDTypeOps
       (-ewise-dtype [_ op xh yh n dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU operations support f16 only" {:dtype dtype*})))
         (let [output (w/-create-buffer-dtype dev n :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :ewise-f16)
                        [xh yh output
                         (wb/uni dev (wb/u32-tag [({:add 0 :sub 1 :mul 2 :div 3} op)
                                                  n 0 0]))]
                        [(wb/ceil-div (wb/ceil-div n 2) 64) 1 1])
           output))
       (-ewise1-dtype [_ op xh n dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU operations support f16 only" {:dtype dtype*})))
         (let [output (w/-create-buffer-dtype dev n :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :ewise1-f16)
                        [xh output
                         (wb/uni dev (wb/u32-tag [({:exp 0 :relu 1 :neg 2 :silu 3
                                                   :sigmoid 4 :tanh 5
                                                   :sigmoid-gradient 6 :tanh-gradient 7
                                                   :gelu 8 :gelu-gradient 9} op)
                                                  n 0 0]))]
                        [(wb/ceil-div (wb/ceil-div n 2) 64) 1 1])
           output))
       (-gemm-dtype [_ Ah m k Bh n dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU operations support f16 only" {:dtype dtype*})))
         (let [output (w/-create-buffer-dtype dev (* m n) :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :gemm-f16)
                        [Ah Bh output (wb/uni dev (wb/u32-tag [m k n 0]))]
                        [(wb/ceil-div (wb/ceil-div (* m n) 2) 64) 1 1])
           output))

       p/IDTypeTensorOps
       (-conv2d-nchw-dtype [_ input-h weight-h bias-h
                            {:keys [n cout oh ow] :as params} dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU convolution supports f16 only" {:dtype dtype*})))
         (let [total (* n cout oh ow)
               output (w/-create-buffer-dtype dev total :storage :f16)
               bias (or bias-h (w/-create-buffer-dtype dev cout :storage :f16))
               values ((juxt :n :cin :h :width :cout :cin-group :kh :kw
                             :oh :ow :sh :sw :ph :pw :dh :dw :groups)
                       params)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :conv2d-nchw-f16)
                        [input-h weight-h bias output
                         (wb/uni dev (wb/u32-tag (into (vec values) [0 0 0])))]
                        [(wb/ceil-div (wb/ceil-div total 2) 64) 1 1])
           output))
       (-group-norm-nchw-dtype [_ input-h weight-h bias-h
                                {:keys [n c h width groups channels-group
                                        group-size eps]} dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU GroupNorm supports f16 only" {:dtype dtype*})))
         (let [total (* n c h width)
               output (w/-create-buffer-dtype dev total :storage :f16)
               weight (or weight-h
                          (let [buffer (w/-create-buffer-dtype dev c :storage :f16)]
                            (w/-write-buffer-dtype dev buffer (repeat c 1.0) :f16)
                            buffer))
               bias (or bias-h (w/-create-buffer-dtype dev c :storage :f16))]
           (w/-dispatch dev (wb/get-pipeline dev pipes :group-norm-nchw-f16)
                        [input-h weight bias output
                         (wb/uni dev (wb/u32-tag
                                      [n c h width groups channels-group
                                       group-size (* h width)]))
                         (wb/uni dev [(double eps)])]
                        [(wb/ceil-div (wb/ceil-div total 2) 64) 1 1])
           output))
       (-embedding-dtype [_ indices-h weight-h {:keys [tokens rows dim]} dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU embedding supports f16 only" {:dtype dtype*})))
         (let [total (* tokens dim)
               output (w/-create-buffer-dtype dev total :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :embedding-f16)
                        [indices-h weight-h output
                         (wb/uni dev (wb/u32-tag [tokens rows dim 0]))]
                        [(wb/ceil-div (wb/ceil-div total 2) 64) 1 1])
           output))
       (-rms-norm-dtype [_ input-h weight-h {:keys [rows dim eps]} dtype*]
         (when-not (and (= dtype* :f16) (even? dim))
           (throw (ex-info "typed GPU RMSNorm requires f16 and even features"
                           {:dtype dtype* :dim dim})))
         (let [output (w/-create-buffer-dtype dev (* rows dim) :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :rms-norm-f16)
                        [input-h weight-h output
                         (wb/uni dev (wb/u32-tag [rows dim 0 0]))
                         (wb/uni dev [(double eps)])]
                        [rows 1 1])
           output))
       (-rotary-embedding-dtype
         [_ input-h {:keys [batch sequence embed heads head-dim position-offset
                            theta direction]} dtype*]
         (when-not (= dtype* :f16)
           (throw (ex-info "typed GPU RoPE supports f16 only" {:dtype dtype*})))
         (let [total (* batch sequence embed)
               output (w/-create-buffer-dtype dev total :storage :f16)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :rotary-embedding-f16)
                        [input-h output
                         (wb/uni dev (wb/u32-tag
                                      [batch sequence embed heads head-dim
                                       position-offset 0 0]))
                         (wb/uni dev [(double theta)])
                         (wb/uni dev [(double direction)])]
                        [(wb/ceil-div (wb/ceil-div total 2) 64) 1 1])
           output))

       p/ITensorBackend
       (-conv2d-nchw [_ input-h weight-h bias-h
                      {:keys [n cin h width cout cin-group kh kw oh ow sh sw ph pw dh dw groups]}]
         (let [total (* n cout oh ow)
               output (w/-create-buffer dev total :storage)
               bias (or bias-h (w/-create-buffer dev cout :storage))
               params [n cin h width cout cin-group kh kw oh ow sh sw ph pw dh dw
                       groups 0 0 0]]
           (let [oc4? (and (= groups 1) (zero? (mod cout 4)))
                 invocations (if oc4? (quot total 4) total)]
             (w/-dispatch
              dev (wb/get-pipeline
                   dev pipes (if oc4? :conv2d-nchw-oc4 :conv2d-nchw))
              [input-h weight-h bias output (wb/uni dev (wb/u32-tag params))]
              [(wb/ceil-div invocations 64) 1 1]))
           output))
       (-group-norm-nchw [_ input-h weight-h bias-h
                          {:keys [n c h width groups channels-group group-size eps]
                           :as params}]
         (let [total (* n c h width)
               output (w/-create-buffer dev total :storage)
               weight (or weight-h
                          (let [buffer (w/-create-buffer dev c :storage)]
                            (w/-write-buffer dev buffer (repeat c 1.0)) buffer))
               bias (or bias-h (w/-create-buffer dev c :storage))
               dims [n c h width groups channels-group group-size (* h width)]]
           (w/-dispatch dev
                        (wb/get-pipeline dev pipes
                                         (if (:silu? params)
                                           :group-norm-silu-nchw
                                           :group-norm-nchw))
                        [input-h weight bias output (wb/uni dev (wb/u32-tag dims))
                         (wb/uni dev [(double eps)])]
                        [(* n groups) 1 1])
           output))
       (-embedding [_ indices-h weight-h {:keys [tokens rows dim]}]
         (let [total (* tokens dim)
               output (w/-create-buffer dev total :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :embedding)
                        [indices-h weight-h output
                         (wb/uni dev (wb/u32-tag [tokens rows dim 0]))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-rms-norm [_ input-h weight-h {:keys [rows dim eps]}]
         (let [output (w/-create-buffer dev (* rows dim) :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :rms-norm)
                        [input-h weight-h output
                         (wb/uni dev (wb/u32-tag [rows dim 0 0]))
                         (wb/uni dev [(double eps)])]
                        [rows 1 1])
           output))
       (-rotary-embedding
         [_ input-h {:keys [batch sequence embed heads head-dim position-offset
                            theta direction]}]
         (let [total (* batch sequence embed)
               output (w/-create-buffer dev total :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :rotary-embedding)
                        [input-h output
                         (wb/uni dev (wb/u32-tag
                                      [batch sequence embed heads head-dim
                                       position-offset 0 0]))
                         (wb/uni dev [(double theta)])
                         (wb/uni dev [(double direction)])]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-upsample-nearest2d [_ input-h
                             {:keys [n c h width oh ow scale-h scale-w]}]
         (let [total (* n c oh ow)
               output (w/-create-buffer dev total :storage)
               dims [n c h width oh ow scale-h scale-w]]
           (w/-dispatch dev (wb/get-pipeline dev pipes :upsample-nearest2d)
                        [input-h output (wb/uni dev (wb/u32-tag dims))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-cat [_ input-handles {:keys [total-output output-block inputs]}]
         (let [output (w/-create-buffer dev total-output :storage)]
           (doseq [[input-h {:keys [total block axis-offset]}]
                   (map vector input-handles inputs)]
             (w/-dispatch dev (wb/get-pipeline dev pipes :cat-copy)
                          [input-h output
                           (wb/uni dev (wb/u32-tag [total block output-block axis-offset]))]
                          [(wb/ceil-div total 64) 1 1]))
           output))
       (-slice-axis [_ input-h {:keys [total input-block output-block input-offset]}]
         (let [output (w/-create-buffer dev total :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :slice-axis)
                        [input-h output
                         (wb/uni dev (wb/u32-tag
                                      [total input-block output-block input-offset]))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-pad-right-bottom-nchw [_ input-h {:keys [total h width output-width]}]
         (let [output (w/-create-buffer dev total :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :pad-right-bottom-nchw)
                        [input-h output
                         (wb/uni dev (wb/u32-tag [total h width output-width]))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-add-last-axis-bias [_ input-h bias-h {:keys [total width]}]
         (let [output (w/-create-buffer dev total :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :add-last-axis-bias)
                        [input-h bias-h output
                         (wb/uni dev (wb/u32-tag [total width 0 0]))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-transpose-2d [_ input-h {:keys [rows cols]}]
         (let [output (w/-create-buffer dev (* rows cols) :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :transpose-2d)
                        [input-h output
                         (wb/uni dev (wb/u32-tag [rows cols]))]
                        [(wb/ceil-div cols 16) (wb/ceil-div rows 16) 1])
           output))
       (-sum-rows [_ input-h {:keys [rows cols]}]
         (let [output (w/-create-buffer dev cols :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :bias-gradient)
                        [input-h output
                         (wb/uni dev (wb/u32-tag [rows cols]))]
                        [(wb/ceil-div cols 64) 1 1])
           output))
       (-mse-loss [_ prediction-h target-h {:keys [count]}]
         (let [loss (w/-create-buffer dev 1 :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :mse-loss)
                        [prediction-h target-h loss
                         (wb/uni dev (wb/u32-tag [count]))]
                        [1 1 1])
           loss))
       (-mse-gradient [_ prediction-h target-h upstream-h {:keys [count]}]
         (let [gradient (w/-create-buffer dev count :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :mse-gradient)
                        [prediction-h target-h upstream-h gradient
                         (wb/uni dev (wb/u32-tag [count]))]
                        [(wb/ceil-div count 64) 1 1])
           gradient))
       (-sgd-step [_ parameter-h gradient-h {:keys [count learning-rate]}]
         (let [output (w/-create-buffer dev count :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :sgd-step)
                        [parameter-h gradient-h output
                         (wb/uni dev [(double learning-rate)])
                         (wb/uni dev (wb/u32-tag [count]))]
                        [(wb/ceil-div count 64) 1 1])
           output))
       (-adamw-step [_ parameter-h gradient-h moment-h variance-h
                     {:keys [count learning-rate beta1 beta2 eps weight-decay
                             correction1 correction2]}]
         (let [moment (or moment-h (w/-create-buffer dev count :storage))
               variance (or variance-h (w/-create-buffer dev count :storage))
               next-parameter (w/-create-buffer dev count :storage)
               next-moment (w/-create-buffer dev count :storage)
               next-variance (w/-create-buffer dev count :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :adamw-step)
                        [parameter-h gradient-h moment variance
                         next-parameter next-moment next-variance
                         (wb/uni dev (mapv double
                                           [learning-rate beta1 beta2 eps
                                            weight-decay correction1 correction2 0.0]))
                         (wb/uni dev (wb/u32-tag [count]))]
                        [(wb/ceil-div count 64) 1 1])
           {:parameter next-parameter :moment next-moment
            :variance next-variance}))
       (-unscale-gradient [_ gradient-h {:keys [count inverse-scale]}]
         (let [output (w/-create-buffer dev count :storage)
               found-inf (w/-create-buffer dev 1 :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :unscale-gradient)
                        [gradient-h output found-inf
                         (wb/uni dev [(double inverse-scale)])
                         (wb/uni dev (wb/u32-tag [count]))]
                        [(wb/ceil-div count 64) 1 1])
           {:gradient output :found-inf found-inf}))
       (-multi-head-attention [_ query-h key-h value-h key-padding-mask-h
                               {:keys [batch seq-q seq-k d-model kv-d-model heads kv-heads head-dim total
                                       causal? has-key-padding-mask?]}]
         (let [output (w/-create-buffer dev total :storage)
               mask (or key-padding-mask-h
                        (w/-create-buffer dev (* batch seq-k) :storage))]
           (w/-dispatch dev (wb/get-pipeline dev pipes :multi-head-attention)
                        [query-h key-h value-h mask output
                         (wb/uni dev (wb/u32-tag
                                      [batch seq-q seq-k d-model kv-d-model
                                       heads kv-heads head-dim total
                                       (if causal? 1 0)
                                       (if has-key-padding-mask? 1 0) 0]))]
                        [(wb/ceil-div total 64) 1 1])
           output))
       (-multi-head-attention-backward [_ query-h key-h value-h key-padding-mask-h
                                        grad-output-h
                                        {:keys [batch seq-q seq-k d-model kv-d-model heads kv-heads head-dim
                                                causal? has-key-padding-mask?]}]
         (let [total-q (* batch seq-q d-model)
               total-k (* batch seq-k kv-d-model)
               total (max total-q total-k)
               mask (or key-padding-mask-h
                        (w/-create-buffer dev (* batch seq-k) :storage))
               grad-query (w/-create-buffer dev total-q :storage)
               grad-key (w/-create-buffer dev total-k :storage)
               grad-value (w/-create-buffer dev total-k :storage)]
           (w/-dispatch dev (wb/get-pipeline dev pipes :multi-head-attention-backward)
                        [query-h key-h value-h mask grad-output-h
                         grad-query grad-key grad-value
                         (wb/uni dev (wb/u32-tag
                                      [batch seq-q seq-k d-model kv-d-model
                                       heads kv-heads head-dim total-q total-k total
                                       (if causal? 1 0)
                                       (if has-key-padding-mask? 1 0) 0 0 0]))]
                        [(wb/ceil-div total 64) 1 1])
           {:query grad-query :key grad-key :value grad-value})))

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
