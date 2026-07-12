(ns num.cuda-test
  (:require [clojure.test :refer [deftest is]]
            [num.contract :as contract] [num.cuda :as cuda]
            [num.protocol :as p]))

(defrecord FakeCudaDriver [next-ptr memory calls]
  cuda/ICudaDriver
  (-device-info [_] {:cuda/device "contract-fake-sm80" :cuda/compute-capability "8.0"
                     :cuda/runtime-version "12.6" :cuda/cublas-version "12.6"
                     :cuda/cusparse-version "12.5"})
  (-malloc-f32 [_ n]
    (let [ptr (swap! next-ptr inc)] (swap! memory assoc ptr (vec (repeat n 0.0))) ptr))
  (-free-ptr [_ ptr] (swap! calls conj :free) (swap! memory dissoc ptr))
  (-h2d-f32 [_ ptr xs] (swap! memory assoc ptr (mapv float xs)))
  (-d2h-f32 [_ ptr n] (subvec (get @memory ptr) 0 n))
  (-cuda-axpy! [_ alpha x y n]
    (swap! calls conj :cublas-saxpy)
    (swap! memory update y #(mapv + % (mapv (partial * alpha) (subvec (get @memory x) 0 n)))))
  (-cuda-scal! [_ alpha x n]
    (swap! calls conj :cublas-sscal)
    (swap! memory update x #(mapv (partial * alpha) (subvec % 0 n))))
  (-cuda-dot [_ x y n]
    (swap! calls conj :cublas-sdot)
    (reduce + (map * (subvec (get @memory x) 0 n) (subvec (get @memory y) 0 n))))
  (-cuda-nrm2 [_ x n]
    (swap! calls conj :cublas-snrm2)
    (Math/sqrt (reduce + (map #(* % %) (subvec (get @memory x) 0 n)))))
  (-cuda-ewise! [_ op x y z n]
    (swap! calls conj :cuda-ewise-kernel)
    (let [f ({:add + :sub - :mul * :div /} op)]
      (swap! memory assoc z (mapv f (subvec (get @memory x) 0 n) (subvec (get @memory y) 0 n)))))
  (-cuda-reduce [_ op x n]
    (swap! calls conj :cub-reduce)
    (reduce ({:sum + :max max :min min} op) (subvec (get @memory x) 0 n)))
  (-cublas-gemv! [_ alpha A m n x beta y]
    (swap! calls conj :cublas-sgemv)
    (let [av (get @memory A) xv (get @memory x) yv (get @memory y)]
      (swap! memory assoc y
             (mapv (fn [i] (+ (* alpha (reduce + (map * (subvec av (* i n) (* (inc i) n)) xv)))
                               (* beta (nth yv i)))) (range m)))))
  (-cublas-gemm! [_ alpha A m k B n beta C]
    (swap! calls conj :cublas-sgemm)
    (let [av (get @memory A) bv (get @memory B) cv (get @memory C)]
      (swap! memory assoc C
             (vec (for [i (range m) j (range n)]
                    (+ (* alpha (reduce + (for [l (range k)]
                                            (* (nth av (+ (* i k) l)) (nth bv (+ (* l n) j))))))
                       (* beta (nth cv (+ (* i n) j)))))))))
  (-cusparse-spmv! [_ csr x y]
    (swap! calls conj :cusparse-spmv)
    (let [xv (get @memory x) rp (:row-ptr csr) ci (:col-idx csr) vals (:vals csr)]
      (swap! memory assoc y
             (mapv (fn [row]
                     (reduce + (for [i (range (aget rp row) (aget rp (inc row)))]
                                 (* (aget vals i) (nth xv (aget ci i))))))
                   (range (:n-rows csr)))))))

(defn fake-driver [] (->FakeCudaDriver (atom 0) (atom {}) (atom [])))

(deftest cuda-backend-satisfies-complete-contract
  (let [driver (fake-driver) backend (cuda/cuda-backend driver)]
    (contract/verify backend (fn [pass? label] (is pass? (str "CUDA contract op: " label))))
    (is (= :cuda (p/-backend-name backend)))
    (is (= "contract-fake-sm80" (:cuda/device (cuda/backend-info backend))))
    (is (every? (set @(:calls driver))
                [:cublas-saxpy :cublas-sscal :cublas-sdot :cublas-snrm2
                 :cublas-sgemv :cublas-sgemm :cusparse-spmv :cuda-ewise-kernel :cub-reduce]))))

(deftest cuda-handle-ownership-and-lifecycle-fail-closed
  (let [a (cuda/cuda-backend (fake-driver)) b (cuda/cuda-backend (fake-driver))
        handle (p/-copy-from-host a [1 2 3])]
    (is (thrown-with-msg? Exception #"foreign" (p/-copy-to-host b handle 3)))
    (p/-free a handle)
    (is (thrown-with-msg? Exception #"released" (p/-copy-to-host a handle 3)))
    (is (thrown-with-msg? Exception #"positive" (p/-alloc a 0)))))

(deftest cuda-driver-provenance-is-mandatory
  (is (thrown-with-msg? Exception #"required provenance"
                        (cuda/cuda-backend
                         (reify cuda/ICudaDriver
                           (-device-info [_] {})
                           (-malloc-f32 [_ _] nil) (-free-ptr [_ _]) (-h2d-f32 [_ _ _]) (-d2h-f32 [_ _ _])
                           (-cuda-axpy! [_ _ _ _ _]) (-cuda-scal! [_ _ _ _]) (-cuda-dot [_ _ _ _] 0)
                           (-cuda-nrm2 [_ _ _] 0) (-cuda-ewise! [_ _ _ _ _ _]) (-cuda-reduce [_ _ _ _] 0)
                           (-cublas-gemv! [_ _ _ _ _ _ _ _]) (-cublas-gemm! [_ _ _ _ _ _ _ _ _])
                           (-cusparse-spmv! [_ _ _ _]))))))
