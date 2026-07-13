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
    (doseq [op [:axpy :scal :ewise :ewise1 :reduce :gemv :gemm :conv2d-nchw :spmv]]
      (is (string? (get w/shaders op)) (str "missing shader: " op))
      (is (re-find #"@compute" (get w/shaders op)) (str op " is not a compute shader"))))
  (testing "the tiled GEMM uses workgroup shared memory (the optimized path)"
    (is (re-find #"var<workgroup>" (:gemm w/shaders)))))

(deftest backend-namespaces-load
  (testing "the WgslBackend constructor and IGpuDevice port compile on the JVM"
    (is (fn? wb/wgsl-backend))
    (is (some? (resolve 'num.wgsl/IGpuDevice)))))
