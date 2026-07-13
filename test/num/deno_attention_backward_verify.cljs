(ns num.deno-attention-backward-verify
  "Verify fused multi-head-attention backward on real Deno WebGPU/Metal.

  The CPU path deliberately uses the decomposed autograd graph, while the GPU
  path dispatches one fused forward kernel and one fused backward kernel. This
  makes agreement an independent cross-implementation check, including the
  less forgiving cross-attention case where seqQ differs from seqK."
  (:require [num.array :as arr]
            [num.autograd :as ag :include-macros true]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]))

(def query-values
  [0.2 -0.1 0.3 0.4
   -0.2 0.1 0.5 -0.3])

(def key-values
  [0.1 0.3 -0.2 0.4
   0.5 -0.1 0.2 0.0
   -0.3 0.2 0.1 0.6])

(def value-values
  [0.4 -0.2 0.1 0.3
   -0.1 0.5 0.2 -0.4
   0.3 0.1 -0.2 0.6])

(def grad-output-values
  [0.3 -0.2 0.5 0.1
   -0.4 0.6 -0.1 0.2])

(def projection-values
  {:qw [0.2 -0.1 0.0 0.3, 0.1 0.4 -0.2 0.0,
        -0.3 0.2 0.5 0.1, 0.0 -0.2 0.1 0.4]
   :kw [0.1 0.0 -0.2 0.3, -0.1 0.5 0.2 0.0,
        0.4 -0.3 0.1 0.2, 0.2 0.1 -0.1 0.3]
   :vw [0.3 -0.2 0.1 0.0, 0.0 0.2 0.4 -0.1,
        -0.2 0.1 0.3 0.5, 0.1 0.0 -0.3 0.2]
   :ow [0.2 0.1 -0.1 0.3, -0.3 0.4 0.2 0.0,
        0.1 -0.2 0.5 0.1, 0.0 0.3 -0.1 0.4]
   :qb [0.01 -0.02 0.03 -0.01]
   :kb [-0.02 0.01 0.0 0.02]
   :vb [0.03 0.0 -0.01 0.01]
   :ob [0.0 0.02 -0.03 0.01]})

(defn- run-graph [backend]
  (let [[graph tape]
        (ag/with-tape
          (let [q (ag/value (arr/from-vec backend query-values [2 4]))
                k (ag/value (arr/from-vec backend key-values [3 4]))
                v (ag/value (arr/from-vec backend value-values [3 4]))]
            {:q q :k k :v v :out (ag/multi-head-attention* q k v 2)}))]
    (ag/backward! (:out graph)
                  (arr/from-vec backend grad-output-values [2 4]) tape)
    {:out (:data (:out graph))
     :query @(:grad (:q graph))
     :key @(:grad (:k graph))
     :value @(:grad (:v graph))}))

(defn- run-projected-graph [backend]
  (let [[graph tape]
        (ag/with-tape
          (let [x (ag/value (arr/from-vec backend query-values [2 4]))
                params (into {}
                             (map (fn [[param-name values]]
                                    [param-name (ag/value
                                           (arr/from-vec backend values
                                                         (if (= 16 (count values))
                                                           [4 4] [4])))])
                                  projection-values))
                project (fn [weight bias]
                          (ag/add-bias* (ag/matmul* x (get params weight))
                                        (get params bias)))
                q (project :qw :qb)
                k (project :kw :kb)
                v (project :vw :vb)
                attention (ag/multi-head-attention* q k v 2)
                out (ag/add-bias* (ag/matmul* attention (:ow params))
                                  (:ob params))]
            {:x x :params params :out out}))]
    (ag/backward! (:out graph)
                  (arr/from-vec backend grad-output-values [2 4]) tape)
    (merge {:projected-out (:data (:out graph))
            :projected-input @(:grad (:x graph))}
           (into {}
                 (map (fn [[param-name parameter]]
                        [(keyword (str "projected-" (name param-name)))
                         @(:grad parameter)])
                      (:params graph))))))

(defn- close? [actual expected]
  (and (= (count actual) (count expected))
       (every? true? (map #(< (js/Math.abs (- %1 %2)) 2.0e-4)
                          actual expected))))

(defn- check! [pass fail label actual expected]
  (let [ok? (close? actual expected)]
    (println (str (if ok? "✓" "✗") " " label
                  (when-not ok?
                    (str "\n  expected=" expected "\n  actual=" actual))))
    (swap! (if ok? pass fail) inc)))

(defn -main [& _]
  (let [cpu-backend (cpu/cpu-backend)
        cpu-result (merge (run-graph cpu-backend)
                          (run-projected-graph cpu-backend))
        expected (into {} (map (fn [[k a]] [k (arr/->vec a)]) cpu-result))]
    (-> (dg/request-device)
        (.then
         (fn [request]
           (println "Fused attention backward on" (dg/adapter-description request))
           (let [gpu (dg/backend request)
                 result (merge (run-graph gpu) (run-projected-graph gpu))
                 pass (atom 0)
                 fail (atom 0)]
             (-> (js/Promise.all
                  (into-array
                   (map (fn [[label a]]
                          (.then (arr/->vec a)
                                 #(check! pass fail (name label) % (get expected label))))
                        result)))
                 (.then (fn [_]
                          (println (str "\nMetal attention forward/backward: "
                                        @pass " passed, " @fail " failed"))
                          (js/Deno.exit (if (zero? @fail) 0 1))))))))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
