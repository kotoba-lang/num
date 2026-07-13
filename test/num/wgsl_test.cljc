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
                :conv2d-nchw :group-norm-nchw :upsample-nearest2d :cat-copy
                :add-last-axis-bias :multi-head-attention
                :multi-head-attention-backward :transpose-2d :bias-gradient
                :mse-loss :mse-gradient :sgd-step :spmv]]
      (is (string? (get w/shaders op)) (str "missing shader: " op))
      (is (re-find #"@compute" (get w/shaders op)) (str op " is not a compute shader"))))
  (testing "the tiled GEMM uses workgroup shared memory (the optimized path)"
    (is (re-find #"var<workgroup>" (:gemm w/shaders))))
  (testing "typed kernels use packed physical halves and accumulate GEMM in f32"
    (is (re-find #"unpack2x16float" (:gemm-f16 w/shaders)))
    (is (re-find #"pack2x16float" (:gemm-f16 w/shaders)))
    (is (re-find #"var sum: f32" (:gemm-f16 w/shaders)))))

(deftest attention-backward-shader-covers-all-three-gradients
  (let [shader (:multi-head-attention-backward w/shaders)]
    (is (re-find #"grad_query" shader))
    (is (re-find #"grad_key" shader))
    (is (re-find #"grad_value" shader))
    (is (re-find #"grad_expectation" shader))))

(deftest attention-shaders-cover-batches-and-masks
  (doseq [shader [(:multi-head-attention w/shaders)
                  (:multi-head-attention-backward w/shaders)]]
    (is (re-find #"key_padding_mask" shader))
    (is (re-find #"p\.causal" shader))
    (is (re-find #"batch \* p\.seq" shader))))

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

(deftest q4-k-gemv-shader-decodes-superblock-scales-and-minima
  (is (string? w/q4-k-gemv-wgsl))
  (is (re-find #"unpack2x16float" w/q4-k-gemv-wgsl))
  (is (re-find #"group_min" w/q4-k-gemv-wgsl)))

(deftest q5-gemv-shaders-restore-fifth-bit
  (is (re-find #"high_bit" w/q5-0-gemv-wgsl))
  (is (re-find #"- 16.0" w/q5-0-gemv-wgsl))
  (is (re-find #"high_bit" w/q5-1-gemv-wgsl))
  (is (re-find #"dm.y" w/q5-1-gemv-wgsl)))

(deftest k-quant-gemv-shaders-cover-five-and-six-bit-superblocks
  (is (re-find #"176u" w/q5-k-gemv-wgsl))
  (is (re-find #"let high" w/q5-k-gemv-wgsl))
  (is (re-find #"210u" w/q6-k-gemv-wgsl))
  (is (re-find #"signed_byte" w/q6-k-gemv-wgsl)))

(deftest persistent-kv-cache-and-causal-gqa-attention-shaders
  (is (re-find #"p.layer \* p.context" w/kv-cache-write-wgsl))
  (is (re-find #"inverseSqrt" w/causal-gqa-attention-wgsl))
  (is (re-find #"maximum" w/causal-gqa-attention-wgsl))
  (is (re-find #"kv_head" w/causal-gqa-attention-wgsl)))

(deftest gpu-training-primitives-cover-forward-backward-and-update
  (is (re-find #"output\[col \* dims.x \+ row\]" w/transpose-2d-wgsl))
  (is (re-find #"prediction\[i\] - expected\[i\]" w/mse-gradient-wgsl))
  (is (re-find #"preactivation\[i\] > 0.0" w/relu-backward-wgsl))
  (is (re-find #"for \(var row" w/bias-gradient-wgsl))
  (is (re-find #"learning_rate \* gradient" w/sgd-update-wgsl)))
