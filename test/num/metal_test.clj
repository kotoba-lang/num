(ns num.metal-test
  (:require [clojure.test :refer [deftest is]] [num.contract :as contract]
            [num.cuda :as cuda] [num.cuda-test :as cuda-test]
            [num.metal :as metal] [num.protocol :as p]))

(defn fake-metal-driver []
  (let [d (cuda-test/fake-driver)]
    {:delegate d
     :driver
     (reify
       metal/IMetalDriver
       (-metal-info [_] {:metal/device "Apple M4" :metal/family "Apple9"
                         :metal/os-version "macOS-26.4" :metal/compiler-version "runtime-msl-v1"})
       (-metal-alloc-f32 [_ n] (cuda/-malloc-f32 d n))
       (-metal-free [_ ptr] (cuda/-free-ptr d ptr))
       (-metal-h2d [_ ptr xs] (cuda/-h2d-f32 d ptr xs))
       (-metal-d2h [_ ptr n] (cuda/-d2h-f32 d ptr n))
       (-metal-axpy! [_ a x y n] (cuda/-cuda-axpy! d a x y n))
       (-metal-scal! [_ a x n] (cuda/-cuda-scal! d a x n))
       (-metal-dot [_ x y n] (cuda/-cuda-dot d x y n))
       (-metal-nrm2 [_ x n] (cuda/-cuda-nrm2 d x n))
       (-metal-ewise! [_ op x y z n] (cuda/-cuda-ewise! d op x y z n))
       (-metal-reduce [_ op x n] (cuda/-cuda-reduce d op x n))
       (-mps-gemv! [_ a A m n x b y] (cuda/-cublas-gemv! d a A m n x b y))
       (-mps-gemm! [_ a A m k B n b C] (cuda/-cublas-gemm! d a A m k B n b C))
       (-metal-spmv! [_ csr x y] (cuda/-cusparse-spmv! d csr x y))
       metal/ICompiledMetalDriver
       (-compile-msl [_ artifact] (cuda/-compile-kernel d artifact))
       (-compiled-metal-ewise! [_ kernel x y z n wg] (cuda/-compiled-ewise! d kernel x y z n wg))
       (-compiled-metal-reduce [_ kernel op x n wg] (cuda/-compiled-reduce d kernel op x n wg)))}))

(deftest metal-backend-satisfies-complete-contract-with-compiled-msl
  (let [{:keys [driver delegate]} (fake-metal-driver) backend (metal/metal-backend driver)]
    (contract/verify backend (fn [pass? label] (is pass? (str "Metal contract op: " label))))
    (is (= :metal (p/-backend-name backend)))
    (is (= "Apple M4" (:metal/device (metal/backend-info backend))))
    (is (= :runtime-msl (:metal/compiler-mode (metal/backend-info backend))))
    (is (= 7 (count (:metal/compiler-artifacts (metal/backend-info backend)))))
    (is (= 7 (count (filter #{:nvrtc-compile} @(:calls delegate)))))
    (is (not-any? #{:cuda-ewise-kernel :cub-reduce} @(:calls delegate)))))

(deftest metal-provenance-and-handle-ownership-fail-closed
  (let [{driver-a :driver} (fake-metal-driver) {driver-b :driver} (fake-metal-driver)
        a (metal/metal-backend driver-a) b (metal/metal-backend driver-b)
        h (p/-copy-from-host a [1 2])]
    (is (thrown-with-msg? Exception #"foreign" (p/-copy-to-host b h 2)))
    (p/-free a h)
    (is (thrown-with-msg? Exception #"released" (p/-copy-to-host a h 2)))
    (is (thrown-with-msg? Exception #"requires IMetalDriver"
                          (metal/metal-backend {})))))
