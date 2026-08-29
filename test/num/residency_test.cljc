(ns num.residency-test
  (:require [clojure.test :refer [deftest is testing]]
            [num.residency :as residency]))

(deftest exact-edge-admission
  (testing "a 16 GiB machine admits a bounded 9B Q4 resident"
    (let [gib 1073741824
          result (residency/admission
                  {:memory-bytes (* 16 gib)
                   :os-reserve-bytes (* 3 gib)
                   :headroom-bytes gib
                   :runtime-bytes 7200000000
                   :context-bytes (* 2 gib)})]
      (is (:admitted? result))
      (is (= (- (* 12 gib) (+ 7200000000 (* 2 gib)))
             (:free-after-bytes result)))))
  (testing "oversubscription fails closed"
    (is (false? (:admitted?
                 (residency/admission
                  {:memory-bytes 10 :os-reserve-bytes 2 :headroom-bytes 1
                   :runtime-bytes 7 :context-bytes 1}))))))

(deftest speculative-memory-is-not-hidden-in-headroom
  (let [base {:memory-bytes 16 :os-reserve-bytes 3 :headroom-bytes 1
              :runtime-bytes 8 :context-bytes 3}]
    (is (:admitted? (residency/admission base)))
    (let [result (residency/admission (assoc base :speculative-bytes 2))]
      (is (false? (:admitted? result)))
      (is (= 2 (:speculative-bytes result)))
      (is (= 13 (:required-bytes result))))))
