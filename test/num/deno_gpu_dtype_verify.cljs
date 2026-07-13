(ns num.deno-gpu-dtype-verify
  "Live shader-f16 verification against the CPU typed-storage oracle."
  (:require [num.array :as arr]
            [num.core :as num]
            [num.cpu :as cpu]
            [num.deno-gpu :as gpu]))

(defn- approx-vec? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) tolerance)
                          expected actual))))

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        cpu-a (arr/from-vec cpu-backend [1.0 2.0 3.0 4.0] [2 2] :f16)
        cpu-b (arr/from-vec cpu-backend [0.5 -1.0 2.0 0.25] [2 2] :f16)
        expected [(arr/->vec (num/add cpu-a cpu-b))
                  (arr/->vec (num/silu cpu-a))
                  (arr/->vec (num/matmul cpu-a cpu-b))]]
    (-> (gpu/request-device)
        (.then
         (fn [device-result]
           (let [backend (gpu/backend device-result)
                 a (arr/from-vec backend [1.0 2.0 3.0 4.0] [2 2] :f16)
                 b (arr/from-vec backend [0.5 -1.0 2.0 0.25] [2 2] :f16)
                 outputs [(num/add a b) (num/silu a) (num/matmul a b)]]
             (println "adapter:" (or (gpu/adapter-description device-result) "unknown"))
             (println "f16 physical bytes:" (.-size (:handle a)))
             (.then
              (js/Promise.all (into-array (map arr/->vec (into [a] outputs))))
              (fn [actual]
                (let [input-values (vec (aget actual 0))
                      actual-values (mapv #(vec (aget actual %)) (range 1 4))
                      _ (println "uploaded:" input-values)
                      _ (println "expected:" expected)
                      _ (println "actual:" actual-values)
                      checks [(= 8 (.-size (:handle a)))
                              (approx-vec? (nth expected 0) (nth actual-values 0) 0.002)
                              (approx-vec? (nth expected 1) (nth actual-values 1) 0.002)
                              (approx-vec? (nth expected 2) (nth actual-values 2) 0.01)]
                      passed (count (filter true? checks))]
                  (println (str "Metal f16: " passed "/" (count checks) " passed"))
                  (when-not (= passed (count checks))
                    (.exit js/Deno 1))))))))
        (.catch
         (fn [error]
           (js/console.error error)
           (.exit js/Deno 1))))))

(set! *main-cli-fn* -main)
