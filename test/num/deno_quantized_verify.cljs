(ns num.deno-quantized-verify
  "Real Metal parity for packed Q4_K matmul without dense weight upload."
  (:require [num.array :as arr]
            [num.cpu :as cpu]
            [num.deno-gpu :as dg]
            [num.dtype :as dtype]
            [num.quantized :as q]))

(def scales [1 2 3 4 49 34 19 60])
(def mins [9 10 11 12 45 30 51 20])
(def packed-scales [193 130 67 196, 137 74 203 76, 209 226 51 76])

(defn half-bytes [value]
  (let [bits (bit-and 0xffff (dtype/f32->f16-bits value))]
    [(bit-and bits 0xff) (bit-and (bit-shift-right bits 8) 0xff)]))

(defn block [d dmin]
  (vec (concat (half-bytes d) (half-bytes dmin) packed-scales
               (repeat 128 0x21))))

(def packed-bytes (vec (concat (block 0.5 0.25) (block -0.125 0.5))))
(def inputs (vec (concat (repeat 256 1.0)
                         (map #(if (even? %) 0.5 -0.25) (range 256)))))

(defn run [backend]
  (q/matmul (arr/from-vec backend inputs [2 256])
            (q/matrix backend packed-bytes [2 256] :q4-k)))

(defn close? [left right]
  (every? #(< (js/Math.abs %) 2.0e-3) (map - left right)))

(defn -main [& _]
  (let [expected (arr/->vec (run (cpu/cpu-backend)))]
    (-> (dg/request-device)
        (.then (fn [request]
                 (println "Packed Q4_K matmul on" (dg/adapter-description request))
                 (arr/->vec (run (dg/backend request)))))
        (.then (fn [actual]
                 (let [ok? (close? actual expected)]
                   (println (str "Q4_K CPU/Metal parity: "
                                 (if ok? "passed" "failed")))
                   (when-not ok?
                     (println "expected=" expected "actual=" actual))
                   (js/Deno.exit (if ok? 0 1)))))
        (.catch (fn [error]
                  (println "ERROR:" (or (.-stack error) (str error)))
                  (js/Deno.exit 1))))))

(set! *main-cli-fn* -main)
