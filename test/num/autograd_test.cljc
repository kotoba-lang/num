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

(deftest conv2d-gradients-match-finite-differences
  (testing "conv2d*'s im2col*/col2im backward (the one genuinely new formula
            — matmul*/reshape* are already verified elsewhere), checked
            against finite differences for BOTH the kernel and the INPUT.
            Checking dL/dx specifically exercises col2im's overlapping-
            window scatter-add (kh=kw=2 on a 4x4 input means every interior
            input pixel contributes to more than one output position) —
            a bug there would not show up if only dL/dkernel were checked.
            All values kept strictly positive (x, kernel, and therefore the
            conv output) to stay clear of relu's non-differentiable kink at
            0, which would make the numerical estimate unstable/misleading,
            not the analytic gradient wrong."
    (let [H 4 W 4 kh 2 kw 2
          xd (arr/from-vec backend (mapv #(* 0.1 (inc %)) (range (* H W))) [H W])
          kd (arr/from-vec backend [0.1 0.2 0.3 0.4] [kh kw])
          targetd (arr/from-vec backend (repeat 9 0.3) [3 3]) ; oh=ow=3 for 4x4 in, 2x2 kernel

          loss-of (fn [xd' kd']
                    (let [[result _tape]
                          (ag/with-tape
                            (let [x (ag/value xd') k (ag/value kd')]
                              (ag/mse-loss* (ag/relu* (ag/conv2d* x k)) targetd)))]
                      (arr/->scalar (:data result))))

          [result tape]
          (ag/with-tape
            (let [x (ag/value xd) k (ag/value kd)
                  loss (ag/mse-loss* (ag/relu* (ag/conv2d* x k)) targetd)]
              {:loss loss :x x :k k}))]
      (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)

      (doseq [[label key data] [["kernel" :k kd] ["input" :x xd]]]
        (testing (str label " gradient")
          (let [analytic (arr/->vec @(:grad (get result key)))
                numeric (arr/->vec
                         (numerical-grad
                          (fn [perturbed]
                            (if (= key :k) (loss-of xd perturbed) (loss-of perturbed kd)))
                          data 1.0e-4))]
            (is (approx-vec-tol? analytic numeric 1.0e-3)
                (str label " analytic=" analytic " numeric=" numeric))))))))

(deftest attention-self-attention-gradient-matches-finite-differences
  (testing "attention*'s transpose*/scale* backward (matmul*/softmax* are
            already verified elsewhere), checked in the SELF-attention
            configuration torch.num-backend's :attention layer actually
            uses — Q=K=V=the SAME Value. This specifically exercises that
            with-tape/backward!'s gradient accumulation correctly SUMS the
            three separate contributions (from x's role as Q, as K via
            transpose*, and as V) into one shared :grad atom, not just that
            each formula is right in isolation."
    (let [seq 3 d 2
          xd (arr/from-vec backend [0.2 -0.1 0.3 0.4 -0.2 0.1] [seq d])
          targetd (arr/from-vec backend (repeat (* seq d) 0.0) [seq d])

          loss-of (fn [xd']
                    (let [[result _tape]
                          (ag/with-tape
                            (let [x (ag/value xd')]
                              (ag/mse-loss* (ag/attention* x x x) targetd)))]
                      (arr/->scalar (:data result))))

          [result tape]
          (ag/with-tape
            (let [x (ag/value xd)
                  loss (ag/mse-loss* (ag/attention* x x x) targetd)]
              {:loss loss :x x}))]
      (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)
      (let [analytic (arr/->vec @(:grad (:x result)))
            numeric (arr/->vec (numerical-grad loss-of xd 1.0e-4))]
        (is (approx-vec-tol? analytic numeric 1.0e-3)
            (str "x analytic=" analytic " numeric=" numeric))))))
(deftest conv2d-nchw-gradients-match-finite-differences
  (testing "batched/channel-aware convolution differentiates input, grouped
            weights, and bias under padding+stride, not only valid 2-D toys"
    (let [xd (arr/from-vec backend (mapv #(* 0.07 (inc %)) (range 18)) [1 2 3 3])
          wd (arr/from-vec backend [0.2 -0.1 0.3 0.4 -0.2 0.5 0.1 -0.3]
                           [2 1 2 2])
          bd (arr/from-vec backend [0.05 -0.08] [2])
          target (arr/from-vec backend [0.2 0.1 -0.1 0.3 0.4 -0.2 0.15 0.05]
                               [1 2 2 2])
          opts {:padding 1 :stride 2 :groups 2}
          loss-of
          (fn [xd' wd' bd']
            (let [[loss _]
                  (ag/with-tape
                    (ag/mse-loss*
                     (ag/conv2d-nchw* (ag/value xd') (ag/value wd') (ag/value bd') opts)
                     target))]
              (arr/->scalar (:data loss))))
          [result tape]
          (ag/with-tape
            (let [x (ag/value xd) weight (ag/value wd) bias (ag/value bd)
                  loss (ag/mse-loss* (ag/conv2d-nchw* x weight bias opts) target)]
              {:loss loss :x x :weight weight :bias bias}))]
      (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)
      (doseq [[label key data]
              [["input" :x xd] ["weight" :weight wd] ["bias" :bias bd]]]
        (let [numeric
              (numerical-grad
               (fn [perturbed]
                 (loss-of (if (= key :x) perturbed xd)
                          (if (= key :weight) perturbed wd)
                          (if (= key :bias) perturbed bd)))
               data 1.0e-5)
              analytic @(:grad (get result key))]
          (is (approx-vec-tol? (arr/->vec analytic) (arr/->vec numeric) 1.0e-4)
              (str label " analytic=" (arr/->vec analytic)
                   " numeric=" (arr/->vec numeric))))))))

(deftest groupnorm-silu-gradients-match-finite-differences
  (testing "UNet normalization+activation differentiates NCHW input and both
            affine parameters, including GroupNorm's coupled group reduction"
    (let [xd (arr/from-vec backend [0.2 -0.4 0.7 1.1 -0.3 0.8 1.4 -0.9]
                           [1 4 2 1])
          wd (arr/from-vec backend [0.8 1.1 -0.7 1.3] [4])
          bd (arr/from-vec backend [0.1 -0.2 0.05 0.3] [4])
          target (arr/from-vec backend [0.0 0.2 -0.1 0.5 0.4 -0.3 0.8 0.1]
                               [1 4 2 1])
          loss-of
          (fn [xd' wd' bd']
            (let [[loss _]
                  (ag/with-tape
                    (ag/mse-loss*
                     (ag/silu*
                      (ag/group-norm-nchw* (ag/value xd') 2
                                           (ag/value wd') (ag/value bd') 1.0e-5))
                     target))]
              (arr/->scalar (:data loss))))
          [result tape]
          (ag/with-tape
            (let [x (ag/value xd) weight (ag/value wd) bias (ag/value bd)
                  normalized (ag/group-norm-nchw* x 2 weight bias 1.0e-5)
                  loss (ag/mse-loss* (ag/silu* normalized) target)]
              {:loss loss :x x :weight weight :bias bias}))]
      (ag/backward! (:loss result) (arr/from-vec backend [1.0] []) tape)
      (doseq [[label key data]
              [["input" :x xd] ["weight" :weight wd] ["bias" :bias bd]]]
        (let [numeric
              (numerical-grad
               (fn [perturbed]
                 (loss-of (if (= key :x) perturbed xd)
                          (if (= key :weight) perturbed wd)
                          (if (= key :bias) perturbed bd)))
               data 1.0e-5)
              analytic @(:grad (get result key))]
          (is (approx-vec-tol? (arr/->vec analytic) (arr/->vec numeric) 1.0e-4)
              (str label " analytic=" (arr/->vec analytic)
                   " numeric=" (arr/->vec numeric))))))))
