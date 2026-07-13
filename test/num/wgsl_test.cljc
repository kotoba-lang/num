(ns num.wgsl-test
  "Loads the GPU backend namespaces (compile check) and asserts the shader set
  matches the IBackend ops. The shaders' RUNTIME correctness on a GPU is verified
  separately on Apple M4 Metal by verify/metal_contract.js (the full contract);
  this test guards that the Clojure side is well-formed and complete on the JVM."
  (:require [clojure.test :refer [deftest is testing]]
            [num.wgsl :as w]
            [num.wgsl-backend :as wb]))

(deftest shader-set-is-complete
  (testing "every accelerated IBackend op has a non-empty WGSL kernel"
    (doseq [op [:axpy :scal :ewise :ewise1 :reduce :gemv :gemm
                :ewise-f16 :ewise1-f16 :gemm-f16
                :conv2d-nchw-f16 :group-norm-nchw-f16
                :conv2d-nchw :group-norm-nchw :upsample-nearest2d :cat-copy :spmv]]
      (is (string? (get w/shaders op)) (str "missing shader: " op))
      (is (re-find #"@compute" (get w/shaders op)) (str op " is not a compute shader"))))
  (testing "the tiled GEMM uses workgroup shared memory (the optimized path)"
    (is (re-find #"var<workgroup>" (:gemm w/shaders))))
  (testing "typed kernels use packed physical halves and accumulate GEMM in f32"
    (is (re-find #"unpack2x16float" (:gemm-f16 w/shaders)))
    (is (re-find #"pack2x16float" (:gemm-f16 w/shaders)))
    (is (re-find #"var sum: f32" (:gemm-f16 w/shaders)))))

(deftest backend-namespaces-load
  (testing "the WgslBackend constructor and IGpuDevice port compile on the JVM"
    (is (fn? wb/wgsl-backend))
    (is (some? (resolve 'num.wgsl/IGpuDevice)))))

(deftest q8-gemv-shader-decodes-packed-signed-blocks
  (is (string? w/q8-0-gemv-wgsl))
  (is (re-find #"signed_byte" w/q8-0-gemv-wgsl))
  (is (re-find #"scale \* signed_byte" w/q8-0-gemv-wgsl)))

(deftest q4-gemv-shader-decodes-low-and-high-nibbles
  (is (string? w/q4-0-gemv-wgsl))
  (is (re-find #"packed & 15u" w/q4-0-gemv-wgsl))
  (is (re-find #"packed >> 4u" w/q4-0-gemv-wgsl)))
