(ns num.deno-gpu-benchmark
  "Reproducible Apple-Metal benchmark for a production-shaped UNet primitive
  chain. Measures forced materialization, not queue submission latency."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]
            [num.tensor :as t]))

(def input-shape [1 32 64 64])
(def weight-shape [64 32 3 3])
(def output-shape [1 64 64 64])

(def input-values
  (mapv #(- (* 0.002 (mod % 257)) 0.25) (range (arr/nelems input-shape))))
(def weight-values
  (mapv #(- (* 0.001 (mod % 97)) 0.048) (range (arr/nelems weight-shape))))
(def bias-values (mapv #(- (* 0.001 %) 0.032) (range 64)))
(def gamma-values (mapv #(+ 0.8 (* 0.002 %)) (range 64)))
(def beta-values (mapv #(- (* 0.001 %) 0.02) (range 64)))

(defn- now [] (.now js/performance))

(defn- tensors [backend]
  {:input (arr/from-vec backend input-values input-shape)
   :weight (arr/from-vec backend weight-values weight-shape)
   :bias (arr/from-vec backend bias-values [64])
   :gamma (arr/from-vec backend gamma-values [64])
   :beta (arr/from-vec backend beta-values [64])})

(defn- chain [{:keys [input weight bias gamma beta]}]
  (-> (t/conv2d-nchw input weight bias {:padding 1})
      (t/group-norm-nchw 8 gamma beta 1.0e-5)
      (t/silu)))

(defn- force-output [output]
  (arr/->vec output))

(defn- full-channel-benchmark [device dtype]
  (let [gpu (dg/backend device)
        input-shape [1 320 64 64]
        weight-shape [320 320 3 3]
        input (arr/from-vec gpu (repeat (arr/nelems input-shape) 0.01)
                            input-shape dtype)
        weight (arr/from-vec gpu (repeat (arr/nelems weight-shape) 0.001)
                             weight-shape dtype)
        bias (arr/from-vec gpu (repeat 320 0.0) [320] dtype)
        run (fn [] (force-output
                    (t/conv2d-nchw input weight bias {:padding 1})))
        cold-start (now)]
    (-> (run)
        (.then
         (fn [cold-values]
           (let [cold-ms (- (now) cold-start)
                 warm-start (now)]
             (-> (run)
                 (.then
                  (fn [warm-values]
                    (let [warm-ms (- (now) warm-start)
                          center (+ (* 160 64 64) (* 32 64) 32)
                          expected (* 320 9 0.01 0.001)]
                      (println "GPU:" (dg/adapter-description device))
                      (println "full channel conv" dtype ":" input-shape "x" weight-shape
                               "->" [1 320 64 64])
                      (println "GPU cold ms:" (.toFixed cold-ms 2))
                      (println "GPU warm ms:" (.toFixed warm-ms 2))
                      (println "center value:" (nth warm-values center)
                               "expected:" expected)
                      (when (or (not= (count cold-values) (* 320 64 64))
                                (> (Math/abs (- (nth warm-values center) expected))
                                   (if (= dtype :f16) 1.0e-3 1.0e-4)))
                        (throw (js/Error. "full-channel convolution check failed")))))))))))))

(defn -main [& args]
  (if (#{"full" "full-f16"} (first args))
    (-> (dg/request-device)
        (.then #(full-channel-benchmark
                 % (if (= "full-f16" (first args)) :f16 :f32)))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) error))
                  (js/Deno.exit 1))))
    (let [cpu-tensors (tensors (cpu/cpu-backend))
        cpu-start (now)
        cpu-output (force-output (chain cpu-tensors))
        cpu-ms (- (now) cpu-start)]
    (-> (dg/request-device)
        (.then
         (fn [device]
           (let [gpu (dg/backend device)
                 gpu-tensors (tensors gpu)
                 cold-start (now)
                 cold-output (force-output (chain gpu-tensors))]
             (.then cold-output
                    (fn [cold-values]
                      (let [cold-ms (- (now) cold-start)
                            warm-start (now)
                            warm-output (force-output (chain gpu-tensors))]
                        (.then warm-output
                               (fn [warm-values]
                                 (let [warm-ms (- (now) warm-start)
                                       max-error
                                       (reduce max 0.0
                                               (map #(Math/abs (- %1 %2))
                                                    cpu-output warm-values))]
                                   (println "GPU:" (dg/adapter-description device))
                                   (println "shape:" input-shape "x" weight-shape
                                            "->" output-shape)
                                   (println "CPU ms:" (.toFixed cpu-ms 2))
                                   (println "GPU cold ms:" (.toFixed cold-ms 2))
                                   (println "GPU warm ms:" (.toFixed warm-ms 2))
                                   (println "warm speedup:" (.toFixed (/ cpu-ms warm-ms) 2) "x")
                                   (println "max abs error:" max-error)
                                   (when (or (not= (count cold-values) (count cpu-output))
                                             (> max-error 1.0e-4))
                                     (throw (js/Error. "GPU benchmark output differs from CPU oracle"))))))))))))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) error))
                  (js/Deno.exit 1)))))))

(set! *main-cli-fn* -main)
