(ns num.autograd-test
  "The correctness bar for autograd is NOT 'runs without error' — it's
  'matches numerical (finite-difference) gradients', the standard technique
  for catching sign errors / wrong Jacobians / off-by-one-axis bugs that a
  merely-doesn't-throw test would miss entirely. Every weight/bias array of a
  linear/relu/linear/softmax/mse-loss network (torch-clj's own README
  architecture, plus a loss) gets its analytic gradient (num.autograd)
  checked element-by-element against a central-difference numerical estimate."
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.autograd :as ag]
            [num.cpu :as cpu]))

(def backend (cpu/cpu-backend))

(defn- approx-vec-tol?
  "Like num.contract/approx-vec?, but with a caller-given tolerance — finite-
  difference gradients carry their own O(eps^2) truncation error, a looser
  and different-in-KIND tolerance than num.contract's fixed 1e-5 (which
  assumes exact-arithmetic backend-vs-backend agreement, not a numerical
  derivative estimate)."
  [u v tol]
  (and (= (count u) (count v))
       (every? true? (map (fn [a b] (< (Math/abs (- (double a) (double b))) tol)) u v))))

(defn- forward-loss
  "Pure forward pass through the SAME ops num.autograd's ops wrap (so a
  discrepancy this test finds is about the BACKWARD formulas, not a
  forward-computation mismatch), returning the scalar loss."
  [xd w1d b1d w2d b2d targetd]
  (let [[result _tape]
        (ag/with-tape
          (let [x (ag/value xd) w1 (ag/value w1d) b1 (ag/value b1d)
                w2 (ag/value w2d) b2 (ag/value b2d)
                h (ag/relu* (ag/add-bias* (ag/matmul* x w1) b1))
                o (ag/softmax* (ag/add-bias* (ag/matmul* h w2) b2))]
            (ag/mse-loss* o targetd)))]
    (arr/->scalar (:data result))))

(defn- numerical-grad
  "Central-difference numerical gradient of `f` (a fn from one NDArray to a
  scalar loss, with every other input closed over) w.r.t. `arrd`, element by
  element. Independent of num.autograd's backward machinery entirely."
  [f arrd eps]
  (let [xs (vec (arr/->vec arrd)) shape (:shape arrd) be (:backend arrd) n (count xs)]
    (arr/from-vec
     be
     (mapv (fn [i]
             (let [plus (arr/from-vec be (assoc xs i (+ (nth xs i) eps)) shape)
                   minus (arr/from-vec be (assoc xs i (- (nth xs i) eps)) shape)]
               (/ (- (f plus) (f minus)) (* 2.0 eps))))
           (range n))
     shape)))

(deftest linear-relu-linear-softmax-mse-gradients-match-finite-differences
  (testing "every weight/bias array's analytic gradient matches a central-
            difference numerical estimate to ~1e-3 (finite-difference's own
            truncation error, not a correctness bug, sets this tolerance —
            tighter than num.contract's usual 1e-5 which assumes exact
            arithmetic, not a numerical derivative estimate)"
    (let [in 3 hidden 4 out 2 batch 2
          xd  (arr/from-vec backend (mapv #(* 0.1 (inc %)) (range (* batch in))) [batch in])
          w1d (arr/from-vec backend (mapv #(- (* 0.05 %) 0.3) (range (* in hidden))) [in hidden])
          b1d (arr/from-vec backend (mapv #(- (* 0.02 %) 0.05) (range hidden)) [hidden])
          w2d (arr/from-vec backend (mapv #(- (* 0.03 %) 0.15) (range (* hidden out))) [hidden out])
          b2d (arr/from-vec backend [0.1 -0.1] [out])
          targetd (arr/from-vec backend [1 0 0 1] [batch out])

          [result tape]
          (ag/with-tape
            (let [x (ag/value xd) w1 (ag/value w1d) b1 (ag/value b1d)
                  w2 (ag/value w2d) b2 (ag/value b2d)
                  h (ag/relu* (ag/add-bias* (ag/matmul* x w1) b1))
                  o (ag/softmax* (ag/add-bias* (ag/matmul* h w2) b2))
                  loss (ag/mse-loss* o targetd)]
              {:loss loss :w1 w1 :b1 b1 :w2 w2 :b2 b2}))]
      (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)

      (doseq [[label key data]
              [["w1" :w1 w1d] ["b1" :b1 b1d] ["w2" :w2 w2d] ["b2" :b2 b2d]]]
        (testing (str label " gradient")
          (let [analytic (arr/->vec @(:grad (get result key)))
                numeric (arr/->vec
                         (numerical-grad
                          (fn [perturbed]
                            (forward-loss xd
                                          (if (= key :w1) perturbed w1d)
                                          (if (= key :b1) perturbed b1d)
                                          (if (= key :w2) perturbed w2d)
                                          (if (= key :b2) perturbed b2d)
                                          targetd))
                          data 1.0e-4))]
            (is (approx-vec-tol? analytic numeric 1.0e-3)
                (str label " analytic=" analytic " numeric=" numeric))))))))

(deftest gradient-descent-actually-reduces-loss
  (testing "not just 'gradients are individually correct' but 'training
            works': ten plain SGD steps (w -= lr*grad) on the same network,
            re-deriving fresh gradients each step, must not increase the
            loss overall — the real, practical proof this is usable for
            training, not only a formula-correctness check"
    (let [in 3 hidden 4 out 2 batch 2 lr 0.5
          targetd (arr/from-vec backend [1 0 0 1] [batch out])
          xd (arr/from-vec backend (mapv #(* 0.1 (inc %)) (range (* batch in))) [batch in])
          step (fn [w1d b1d w2d b2d]
                 (let [[result tape]
                       (ag/with-tape
                         (let [x (ag/value xd) w1 (ag/value w1d) b1 (ag/value b1d)
                               w2 (ag/value w2d) b2 (ag/value b2d)
                               h (ag/relu* (ag/add-bias* (ag/matmul* x w1) b1))
                               o (ag/softmax* (ag/add-bias* (ag/matmul* h w2) b2))
                               loss (ag/mse-loss* o targetd)]
                           {:loss loss :w1 w1 :b1 b1 :w2 w2 :b2 b2}))]
                   (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)
                   (let [descend (fn [param-data grad-value]
                                    (arr/from-vec backend
                                                  (map (fn [p g] (- p (* lr g)))
                                                       (arr/->vec param-data)
                                                       (arr/->vec @(:grad grad-value)))
                                                  (:shape param-data)))]
                     {:loss (arr/->scalar (:data (:loss result)))
                      :w1 (descend w1d (:w1 result)) :b1 (descend b1d (:b1 result))
                      :w2 (descend w2d (:w2 result)) :b2 (descend b2d (:b2 result))})))]
      (loop [n 10
             w1d (arr/from-vec backend (mapv #(- (* 0.05 %) 0.3) (range (* in hidden))) [in hidden])
             b1d (arr/from-vec backend (mapv #(- (* 0.02 %) 0.05) (range hidden)) [hidden])
             w2d (arr/from-vec backend (mapv #(- (* 0.03 %) 0.15) (range (* hidden out))) [hidden out])
             b2d (arr/from-vec backend [0.1 -0.1] [out])
             losses []]
        (if (zero? n)
          (do
            (is (> (count losses) 1))
            (is (< (last losses) (first losses))
                (str "losses: " losses)))
          (let [{:keys [loss w1 b1 w2 b2]} (step w1d b1d w2d b2d)]
            (recur (dec n) w1 b1 w2 b2 (conj losses loss))))))))
