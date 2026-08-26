(ns num.deno-sampling-benchmark
  "Device-resident selection microbenchmark at a production-sized vocabulary."
  (:require [num.array :as arr]
            [num.deno-gpu :as gpu]))

(defn- percentile [values fraction]
  (let [values (vec (sort values))
        index (min (dec (count values))
                   (long (Math/floor (* fraction (count values)))))]
    (nth values index)))

(defn- timed [f]
  (let [started (.now js/performance)]
    (-> (js/Promise.resolve (f))
        (.then (fn [value]
                 {:value value :ms (- (.now js/performance) started)})))))

(defn- measure [iterations f]
  (-> (js/Promise.resolve (f))
      (.then
       (fn [_]
         (reduce (fn [promise _]
                   (.then promise
                          (fn [samples]
                            (-> (timed f)
                                (.then #(conj samples %))))))
                 (js/Promise.resolve []) (range iterations))))
      (.then
       (fn [samples]
         (let [times (mapv :ms samples)]
           {:token (:value (last samples))
            :iterations iterations
            :median-ms (percentile times 0.5)
            :min-ms (apply min times)
            :max-ms (apply max times)})))))

(defn- run-benchmarks [request backend logits draft vocab iterations baseline]
  (let [sample #(arr/sample-top-k-row
                 logits 0 {:top-k % :temperature 0.7
                           :top-p 0.9 :random-value 0.51})]
    (-> (measure iterations #(sample 8))
        (.then (fn [k8]
                 (-> (measure iterations #(sample 40))
                     (.then (fn [k40] [k8 k40])))))
        (.then (fn [[k8 k40]]
                 (-> (measure iterations #(sample 256))
                     (.then (fn [k256] [k8 k40 k256])))))
        (.then
         (fn [[k8 k40 k256]]
           (-> (measure iterations
                        #(arr/sample-softmax-row
                          logits 0 {:temperature 0.7 :random-value 0.51}))
               (.then #(vector k8 k40 k256 %)))))
        (.then
         (fn [[k8 k40 k256 full-softmax]]
           (-> (measure iterations
                        #(arr/speculative-rejection-row
                          logits draft 0 17
                          {:temperature 0.7 :acceptance-random 0.9
                           :residual-random 0.51}))
               (.then #(vector k8 k40 k256 full-softmax %)))))
        (.then
         (fn [[k8 k40 k256 full-softmax speculative]]
           (let [after (gpu/backend-stats backend)
                 result {:adapter (gpu/adapter-description request)
                         :vocab vocab
                         :top-k-8 k8 :top-k-40 k40 :top-k-256 k256
                         :full-softmax full-softmax
                         :speculative speculative
                         :readbacks (- (:selection-readbacks after 0)
                                       (:selection-readbacks baseline 0))
                         :readback-bytes
                         (- (:selection-readback-bytes after 0)
                            (:selection-readback-bytes baseline 0))}]
             (arr/release! logits)
             (arr/release! draft)
             (println (js/JSON.stringify (clj->js result)))))))))

(defn -main [& _]
  (let [vocab 262144 iterations 3
        values (mapv #(Math/sin (* 0.017 %)) (range vocab))
        draft-values (mapv #(Math/cos (* 0.013 %)) (range vocab))]
    (-> (gpu/request-device)
        (.then (fn [request]
                 (let [backend (gpu/backend request)
                       logits (arr/from-vec backend values [1 vocab])
                       draft (arr/from-vec backend draft-values [1 vocab])]
                   (run-benchmarks request backend logits draft vocab iterations
                                   (gpu/backend-stats backend)))))
        (.then (fn [_] (js/Deno.exit 0)))
        (.catch (fn [error]
                  (js/console.error (or (.-stack error) error))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
