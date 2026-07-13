(ns num.deno-q8-worker
  "Persistent JSON-lines Q8_0 GEMV worker backed by WebGPU/Metal."
  (:require [num.wgsl :as wgsl]))

(def usage (.-GPUBufferUsage js/globalThis))
(def map-mode (.-GPUMapMode js/globalThis))
(def readline (js/require "node:readline"))
(def registry (atom {}))

(defn- gpu-buffer [device typed-array flags]
  (let [buffer (.createBuffer device #js {:size (max 4 (.-byteLength typed-array))
                                           :usage flags :mappedAtCreation true})
        constructor (.-constructor typed-array)
        mapped (new constructor (.getMappedRange buffer))]
    (.set mapped typed-array) (.unmap buffer) buffer))

(defn- half [bits]
  (let [sign (if (zero? (bit-and bits 0x8000)) 1 -1)
        exponent (bit-and (unsigned-bit-shift-right bits 10) 31)
        fraction (bit-and bits 1023)]
    (cond (zero? exponent) (* sign (Math/pow 2 -14) (/ fraction 1024))
          (= exponent 31) (if (zero? fraction) (* sign js/Infinity) js/NaN)
          :else (* sign (Math/pow 2 (- exponent 15)) (+ 1 (/ fraction 1024))))))

(defn- decode-base64 [text]
  (js/Uint8Array.from (js/Buffer.from text "base64")))

(defn- reply [value] (.write (.-stdout js/process) (str (js/JSON.stringify (clj->js value)) "\n")))

(defn- upload! [device pipeline command]
  (let [id (aget command "id") rows (aget command "rows") cols (aget command "cols")
        blocks (/ cols 32) source (decode-base64 (aget command "data"))
        qbytes (js/Uint8Array. (* rows blocks 32)) scales (js/Float32Array. (* rows blocks))]
    (dotimes [block (* rows blocks)]
      (let [src (* block 34) bits (bit-or (aget source src)
                                          (bit-shift-left (aget source (inc src)) 8))]
        (aset scales block (half bits))
        (.set qbytes (.subarray source (+ src 2) (+ src 34)) (* block 32))))
    (let [storage (bit-or (.-STORAGE usage) (.-COPY_DST usage))
          qbuf (gpu-buffer device (js/Uint32Array. (.-buffer qbytes)) storage)
          sbuf (gpu-buffer device scales storage)
          params (gpu-buffer device (js/Uint32Array. #js [rows cols blocks 0])
                             (bit-or (.-UNIFORM usage) (.-COPY_DST usage)))]
      (swap! registry assoc id #js {:q qbuf :scales sbuf :params params
                                    :rows rows :cols cols :pipeline pipeline})
      (reply {:ok true :id id :gpu-bytes (+ (.-size qbuf) (.-size sbuf))}))))

(defn- gemv! [device command]
  (let [id (aget command "id") entry (get @registry id)]
    (when-not entry (throw (js/Error. (str "unknown Q8 handle: " id))))
    (let [rows (.-rows entry) cols (.-cols entry)
          values (js/Float32Array. (aget command "x"))
          _ (when-not (= cols (.-length values))
              (throw (js/Error. "GEMV input length mismatch")))
          storage (bit-or (.-STORAGE usage) (.-COPY_DST usage))
          xbuf (gpu-buffer device values storage)
          ybuf (.createBuffer device #js {:size (* rows 4)
                                           :usage (bit-or (.-STORAGE usage) (.-COPY_SRC usage))})
          read (.createBuffer device #js {:size (* rows 4)
                                           :usage (bit-or (.-COPY_DST usage) (.-MAP_READ usage))})
          bind (.createBindGroup device
                                 #js {:layout (.getBindGroupLayout (.-pipeline entry) 0)
                                      :entries #js [#js {:binding 0 :resource #js {:buffer (.-q entry)}}
                                                    #js {:binding 1 :resource #js {:buffer (.-scales entry)}}
                                                    #js {:binding 2 :resource #js {:buffer xbuf}}
                                                    #js {:binding 3 :resource #js {:buffer ybuf}}
                                                    #js {:binding 4 :resource #js {:buffer (.-params entry)}}]})
          encoder (.createCommandEncoder device) pass (.beginComputePass encoder)]
      (.setPipeline pass (.-pipeline entry)) (.setBindGroup pass 0 bind)
      (.dispatchWorkgroups pass (Math/ceil (/ rows 64))) (.end pass)
      (.copyBufferToBuffer encoder ybuf 0 read 0 (* rows 4))
      (.submit (.-queue device) #js [(.finish encoder)])
      (-> (.mapAsync read (.-READ map-mode))
          (.then (fn []
                   (let [output (vec (js/Float32Array. (.getMappedRange read)))]
                     (reply {:ok true :id id :y output})
                     (.destroy xbuf) (.destroy ybuf) (.destroy read))))))))

(defn- release! [command]
  (let [id (aget command "id") entry (get @registry id)]
    (when entry
      (.destroy (.-q entry)) (.destroy (.-scales entry)) (.destroy (.-params entry))
      (swap! registry dissoc id))
    (reply {:ok true :id id})))

(defn- serve! [device]
  (let [module (.createShaderModule device #js {:code wgsl/q8-0-gemv-wgsl})
        pipeline (.createComputePipeline device
                                         #js {:layout "auto"
                                              :compute #js {:module module :entryPoint "main"}})
        lines (.createInterface readline #js {:input (.-stdin js/process)})]
    (.on lines "line"
         (fn [line]
           (try
             (let [command (js/JSON.parse line) op (aget command "op")]
               (case op
                 "upload-q8" (upload! device pipeline command)
                 "gemv" (.catch (gemv! device command)
                                 #(reply {:ok false :error (.-message %)}))
                 "release" (release! command)
                 "stats" (reply {:ok true :handles (count @registry)})
                 (reply {:ok false :error (str "unknown op: " op)})))
             (catch :default error
               (reply {:ok false :error (.-message error)})))))))

(defn -main [& _]
  (-> (.requestAdapter (.-gpu js/navigator))
      (.then (fn [adapter]
               (when-not adapter (throw (js/Error. "no WebGPU adapter")))
               (.then (.requestDevice adapter) serve!)))
      (.catch (fn [error] (js/console.error error) (.exit js/Deno 1)))))

(set! *main-cli-fn* -main)
