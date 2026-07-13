(ns num.quantized-test
  (:require [clojure.test :refer [deftest is]]
            [num.array :as arr]
            [num.contract :as contract]
            [num.core :as nm]
            [num.cpu :as cpu]
            [num.dtype :as dtype]
            [num.quantized :as q]))

(defn- half-bytes [value]
  (let [bits (bit-and 0xffff (dtype/f32->f16-bits value))]
    [(bit-and bits 0xff) (bit-and (bit-shift-right bits 8) 0xff)]))

(def scales [1 2 3 4 49 34 19 60])
(def mins [9 10 11 12 45 30 51 20])
(def packed-scales [193 130 67 196, 137 74 203 76, 209 226 51 76])

(defn- block [d dmin]
  (vec (concat (half-bytes d) (half-bytes dmin) packed-scales
               (repeat 128 0x21))))

(defn- dense-row [d dmin]
  (vec (mapcat (fn [index]
                 (let [quant (if (even? index) 1 2)]
                   (repeat 32 (- (* d (nth scales index) quant)
                                 (* dmin (nth mins index))))))
               (range 8))))

(deftest q4-k-matmul-matches-dense-oracle-without-expanding-weight
  (let [backend (cpu/cpu-backend)
        bytes (vec (concat (block 0.5 0.25) (block -0.125 0.5)))
        weight (q/matrix backend bytes [2 256] :q4-k)
        input-values (vec (concat (repeat 256 1.0)
                                  (map #(if (even? %) 0.5 -0.25) (range 256))))
        input (arr/from-vec backend input-values [2 256])
        dense-values (vec (mapcat (fn [inner]
                                    [(nth (dense-row 0.5 0.25) inner)
                                     (nth (dense-row -0.125 0.5) inner)])
                                  (range 256)))
        dense (arr/from-vec backend dense-values [256 2])]
    (is (= [256 2] (:shape weight)))
    (is (= 288 (:byte-count weight)))
    (is (< (:byte-count weight) (* 4 (arr/nelems (:shape weight)))))
    (is (contract/approx-vec? (arr/->vec (nm/matmul input dense))
                              (arr/->vec (q/matmul input weight))))))
