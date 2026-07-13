(ns num.dtype-test
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.core :as num]
            [num.dtype :as dtype]))

(def backend (cpu/cpu-backend))

(deftest physical-sixteen-bit-storage-roundtrips
  (doseq [[dtype* tolerance] [[:f16 5.0e-4] [:bf16 4.0e-3]]]
    (testing (name dtype*)
      (let [a (arr/from-vec backend [1.0 -0.3333 0.0001 65504.0] [4] dtype*)
            values (arr/->vec a)]
        (is (= dtype* (:dtype a)))
        (is (= 2 (dtype/element-bytes dtype*)))
        (is (= 8 (* (alength (:data (:handle a)))
                    (dtype/element-bytes dtype*))))
        (is (< (Math/abs (- -0.3333 (second values))) tolerance))
        (is (= dtype* (:dtype (arr/like a))))))))

(deftest casts-materialize-and-quantize
  (let [source (arr/from-vec backend [0.1 0.2 0.3] [3])
        half (arr/cast source :f16)
        restored (arr/cast half :f32)]
    (is (= :f16 (:dtype half)))
    (is (= :f32 (:dtype restored)))
    (is (not (identical? (:handle source) (:handle half))))
    (is (every? #(< % 2.0e-4)
                (map #(Math/abs (- %1 %2))
                     (arr/->vec source) (arr/->vec restored))))))

(deftest special-values-survive-sixteen-bit-storage
  (doseq [dtype* [:f16 :bf16]]
    (let [[positive negative nan] (arr/->vec
                                   (arr/from-vec backend
                                                 [##Inf ##-Inf ##NaN]
                                                 [3] dtype*))]
      (is (= ##Inf positive))
      (is (= ##-Inf negative))
      (is #?(:clj (Double/isNaN (double nan))
             :cljs (js/isNaN nan))))))

(deftest typed-elementwise-activation-and-gemm
  (doseq [dtype* [:f16 :bf16]]
    (let [a (arr/from-vec backend [1.0 2.0 3.0 4.0] [2 2] dtype*)
          b (arr/from-vec backend [0.5 -1.0 2.0 0.25] [2 2] dtype*)
          sum (num/add a b)
          activated (num/silu sum)
          product (num/matmul a b)]
      (is (= dtype* (:dtype sum) (:dtype activated) (:dtype product)))
      (is (= [1.5 1.0 5.0 4.25] (arr/->vec sum)))
      (is (< (Math/abs (- 4.5 (first (arr/->vec product)))) 0.02))
      (is (< (Math/abs (- -0.5 (second (arr/->vec product)))) 0.02)))))
