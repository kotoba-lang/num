(ns num.gpu-compiler-test
  (:require [clojure.test :refer [deftest is]] [num.gpu-compiler :as gpu-compiler]))

(deftest all-num-generated-kernels-are-sealed-and-cross-target-identical
  (let [registry (gpu-compiler/artifact-registry)]
    (is (= 21 (count registry)))
    (doseq [[kind operator] (keys gpu-compiler/kernel-specs)]
      (let [wgsl (get registry [kind operator :wgsl-v1])
            cuda (get registry [kind operator :cuda-v1])
            msl (get registry [kind operator :msl-v1])]
        (is (= (:kir-sha256 wgsl) (:kir-sha256 cuda)))
        (is (= (:kir-sha256 wgsl) (:kir-sha256 msl)))
        (is (not= (:code-sha256 wgsl) (:code-sha256 cuda)))
        (is (re-find #"@compute" (:code wgsl)))
        (is (re-find #"__global__" (:code cuda)))
        (is (re-find #"metal_stdlib" (:code msl)))))))

(deftest unknown-kernel-is-rejected
  (is (thrown-with-msg? Exception #"unsupported compiled numerical kernel"
                        (gpu-compiler/compile-kernel :fft :forward :cuda-v1))))
