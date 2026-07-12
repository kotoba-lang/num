(ns num.solver
  "Backend-neutral iterative solvers. Vectors remain on their IBackend between
  iterations; only convergence scalars cross to the host."
  (:require [num.array :as arr] [num.core :as num] [num.protocol :as p]))

(defn- release! [a]
  (when a (p/-free (:backend a) (:handle a))))

(defn pcg
  "Conjugate-gradient solve A*x=b for symmetric positive-definite CSR A.
  Returns x plus convergence/provenance diagnostics. This initial variant uses
  the identity preconditioner; the API name leaves room for injected Jacobi/
  AMG preconditioners without changing consumers."
  [backend csr rhs {:keys [tolerance max-iterations]
                    :or {tolerance 1.0e-6 max-iterations 1000}}]
  (let [n (:n-rows csr)]
    (when-not (and (p/backend? backend) (= n (:n-cols csr)) (= [n] (:shape rhs))
                   (identical? backend (:backend rhs)) (pos? tolerance) (pos-int? max-iterations))
      (throw (ex-info "invalid PCG system or options" {:rows n :cols (:n-cols csr)
                                                        :rhs-shape (:shape rhs)})))
    (let [x (arr/from-vec backend (repeat n 0.0) [n])
          ax (num/spmv backend csr x)
          r (num/sub rhs ax)
          zero (arr/from-vec backend (repeat n 0.0) [n])
          direction (num/add r zero)
          initial-r2 (double (num/dot r r))
          target2 (* tolerance tolerance (max 1.0 initial-r2))]
      (release! ax) (release! zero)
      (loop [iteration 0 r2 initial-r2]
        (if (or (<= r2 target2) (= iteration max-iterations))
          (let [status (if (<= r2 target2) :converged :max-iterations)
                result {:solver/status status :solver/iterations iteration
                        :solver/residual-norm (#?(:clj Math/sqrt :cljs js/Math.sqrt) r2)
                        :solver/relative-residual (if (zero? initial-r2) 0.0
                                                    (#?(:clj Math/sqrt :cljs js/Math.sqrt)
                                                     (/ r2 initial-r2)))
                        :solver/backend (p/-backend-name backend) :solver/x x}]
            (release! r) (release! direction) result)
          (let [ap (num/spmv backend csr direction)
                denominator (double (num/dot direction ap))]
            (when (<= denominator 0.0)
              (release! ap) (release! r) (release! direction) (release! x)
              (throw (ex-info "PCG matrix is not positive definite"
                              {:iteration iteration :denominator denominator})))
            (let [alpha (/ r2 denominator)]
              (num/axpy! alpha direction x)
              (num/axpy! (- alpha) ap r)
              (release! ap)
              (let [next-r2 (double (num/dot r r)) beta (/ next-r2 r2)]
                (num/scal! beta direction)
                (num/axpy! 1.0 r direction)
                (recur (inc iteration) next-r2)))))))))
