(ns num.deno-gpu-dtype-verify
  "Live shader-f16 verification against the CPU typed-storage oracle."
  (:require [num.array :as arr]
            [num.core :as num]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]
            [num.tensor :as tensor]))

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        cpu-a (arr/from-vec cpu-backend [1.0 2.0 3.0 4.0] [2 2] :f16)
        cpu-b (arr/from-vec cpu-backend [0.5 -1.0 2.0 0.25] [2 2] :f16)
        conv-input-values (mapv #(- (* 0.03 %) 0.4) (range 32))
        conv-weight-values (mapv #(- (* 0.02 (mod % 17)) 0.12) (range 72))
        conv-bias-values [0.01 -0.02 0.03 -0.04]
        cpu-conv (tensor/conv2d-nchw
                  (arr/from-vec cpu-backend conv-input-values [1 2 4 4] :f16)
                  (arr/from-vec cpu-backend conv-weight-values [4 2 3 3] :f16)
                  (arr/from-vec cpu-backend conv-bias-values [4] :f16)
                  {:padding 1})
        cpu-norm (tensor/group-norm-nchw
                  cpu-conv 2
                  (arr/from-vec cpu-backend [0.9 1.0 1.1 1.2] [4] :f16)
                  (arr/from-vec cpu-backend [0.01 -0.02 0.03 -0.04] [4] :f16)
                  1.0e-5)
        expected [(arr/->vec (num/add cpu-a cpu-b))
                  (arr/->vec (num/silu cpu-a))
                  (arr/->vec (num/sigmoid cpu-a))
                  (arr/->vec (num/tanh cpu-a))
                  (arr/->vec (num/gelu cpu-a))
                  (arr/->vec (num/matmul cpu-a cpu-b))
                  (arr/->vec cpu-conv)
                  (arr/->vec cpu-norm)]]
    (-> (gpu/request-device)
        (.then
         (fn [device-result]
           (let [backend (gpu/backend device-result)
                 a (arr/from-vec backend [1.0 2.0 3.0 4.0] [2 2] :f16)
                 b (arr/from-vec backend [0.5 -1.0 2.0 0.25] [2 2] :f16)
                 conv (tensor/conv2d-nchw
                       (arr/from-vec backend conv-input-values [1 2 4 4] :f16)
                       (arr/from-vec backend conv-weight-values [4 2 3 3] :f16)
                       (arr/from-vec backend conv-bias-values [4] :f16)
                       {:padding 1})
                 norm (tensor/group-norm-nchw
                       conv 2
                       (arr/from-vec backend [0.9 1.0 1.1 1.2] [4] :f16)
                       (arr/from-vec backend [0.01 -0.02 0.03 -0.04] [4] :f16)
                       1.0e-5)
                 outputs [(num/add a b) (num/silu a) (num/sigmoid a) (num/tanh a)
                          (num/gelu a)
                          (num/matmul a b) conv norm]]
             (println "adapter:" (or (gpu/adapter-description device-result) "unknown"))
             (println "f16 physical bytes:" (.-size (:handle a)))
             (.then
              (js/Promise.all (into-array (map arr/->vec (into [a] outputs))))
              (fn [actual]
                (let [input-values (vec (aget actual 0))
                      actual-values (mapv #(vec (aget actual %)) (range 1 9))
                      _ (println "uploaded:" input-values)
                      checks [(= 8 (.-size (:handle a)))
                              (approx-vec? (nth expected 0) (nth actual-values 0) 0.002)
                              (approx-vec? (nth expected 1) (nth actual-values 1) 0.002)
                              (approx-vec? (nth expected 2) (nth actual-values 2) 0.002)
                              (approx-vec? (nth expected 3) (nth actual-values 3) 0.002)
                              (approx-vec? (nth expected 4) (nth actual-values 4) 0.002)
                              (approx-vec? (nth expected 5) (nth actual-values 5) 0.01)
                              (approx-vec? (nth expected 6) (nth actual-values 6) 0.01)
                              (approx-vec? (nth expected 7) (nth actual-values 7) 0.03)]
                      passed (count (filter true? checks))]
                  (println (str "Metal f16: " passed "/" (count checks) " passed"))
                  (when-not (= passed (count checks))
                    (.exit js/Deno 1))))))))
        (.catch
         (fn [error]
           (js/console.error error)
           (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
