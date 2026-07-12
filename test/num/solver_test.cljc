(ns num.solver-test
  (:require [clojure.test :refer [deftest is]] [num.array :as arr]
            [num.cpu :as cpu] [num.solver :as solver] [num.sparse :as sparse]))

(deftest pcg-solves-spd-poisson-system-and-reports-provenance
  (let [backend (cpu/cpu-backend)
        A (sparse/csr 3 [[[0 2.0] [1 -1.0]]
                         [[0 -1.0] [1 2.0] [2 -1.0]]
                         [[1 -1.0] [2 2.0]]])
        rhs (arr/from-vec backend [1 0 1] [3])
        result (solver/pcg backend A rhs {:tolerance 1.0e-12 :max-iterations 20})]
    (is (= :converged (:solver/status result)))
    (is (= :cpu (:solver/backend result)))
    (is (<= (:solver/iterations result) 3))
    (is (< (:solver/residual-norm result) 1.0e-12))
    (is (every? #(< (Math/abs (- % 1.0)) 1.0e-12) (arr/->vec (:solver/x result))))))

(deftest pcg-rejects-nonsymmetric-positive-definite-failure
  (let [backend (cpu/cpu-backend) A (sparse/csr 1 [[[0 -1.0]]])
        rhs (arr/from-vec backend [1] [1])]
    (is (thrown-with-msg? #?(:clj Exception :cljs js/Error) #"not positive definite"
                          (solver/pcg backend A rhs {})))))
