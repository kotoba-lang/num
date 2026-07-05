(ns num.tensor-test
  "Phase 1 (ADR-2607051400 §Phase 1) N-D tensor correctness tests, following this
  repo's existing convention (`num.contract`/`num.cpu-test`): every op is checked
  against a hand-computed or independently-written reference value on the CPU
  oracle backend — never against NumPy (no such dependency exists or ever will)."
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.core :as nm]
            [num.cpu :as cpu]
            [num.tensor :as t]))

(def backend (cpu/cpu-backend))

;; --- broadcast-shapes / broadcast-to -------------------------------------------

(deftest broadcast-shapes-rules
  (testing "equal shapes"
    (is (= [2 3] (t/broadcast-shapes [2 3] [2 3]))))
  (testing "trailing-aligned rank mismatch (scalar vs vector)"
    (is (= [4] (t/broadcast-shapes [] [4])))
    (is (= [4] (t/broadcast-shapes [4] []))))
  (testing "size-1 stretch, row × column outer-product style"
    (is (= [3 4] (t/broadcast-shapes [3 1] [1 4])))
    (is (= [2 3 4] (t/broadcast-shapes [3 4] [2 1 4]))))
  (testing "incompatible non-1 sizes error"
    (is (thrown? #?(:clj Exception :cljs js/Error) (t/broadcast-shapes [2 3] [2 4])))))

(deftest broadcast-to-scalar-plus-vector
  (testing "a 0-D scalar broadcasts to every element of a vector"
    (let [s (arr/from-vec backend [7] [])
          v (arr/->vec (t/broadcast-to s [4]))]
      (is (= [7.0 7.0 7.0 7.0] v)))))

(deftest broadcast-to-row-column-outer
  (testing "[3 1] broadcast to [3 4] repeats each row's single value across columns"
    (let [col (arr/from-vec backend [1 2 3] [3 1])
          out (arr/->vec (t/broadcast-to col [3 4]))]
      (is (= [1.0 1.0 1.0 1.0
              2.0 2.0 2.0 2.0
              3.0 3.0 3.0 3.0] out))))
  (testing "[1 4] broadcast to [3 4] repeats the single row across all rows"
    (let [row (arr/from-vec backend [10 20 30 40] [1 4])
          out (arr/->vec (t/broadcast-to row [3 4]))]
      (is (= [10.0 20.0 30.0 40.0
              10.0 20.0 30.0 40.0
              10.0 20.0 30.0 40.0] out)))))

(deftest add-sub-mul-div-broadcast
  (testing "scalar + vector"
    (let [s (arr/from-vec backend [10] [])
          v (arr/from-vec backend [1 2 3] [3])]
      (is (= [11.0 12.0 13.0] (arr/->vec (t/add s v))))
      (is (= [-9.0 -8.0 -7.0] (arr/->vec (t/sub v s))))
      (is (= [10.0 20.0 30.0] (arr/->vec (t/mul v s))))
      (is (= [0.1 0.2 0.3] (arr/->vec (t/div v s))))))
  (testing "outer-product-style row × column add ([3 1] + [1 4] -> [3 4])"
    (let [col (arr/from-vec backend [1 2 3] [3 1])
          row (arr/from-vec backend [10 20 30 40] [1 4])
          out (arr/->vec (t/add col row))
          ;; independently-derived reference: out[i][j] = col[i] + row[j]
          expect (vec (for [i (range 3) j (range 4)]
                        (double (+ (nth [1 2 3] i) (nth [10 20 30 40] j)))))]
      (is (= expect out))))
  (testing "equal shapes still work exactly as num.core (no behavior change)"
    (let [a (arr/from-vec backend [1 2 3 4] [4])
          b (arr/from-vec backend [4 3 2 1] [4])]
      (is (= (arr/->vec (nm/add a b)) (arr/->vec (t/add a b))))))
  (testing "incompatible shapes error"
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (t/add (arr/from-vec backend [1 2 3] [3]) (arr/from-vec backend [1 2] [2]))))))

;; --- reshape --------------------------------------------------------------------

(deftest reshape-preserves-data-layout
  (testing "[2 6] -> [3 4] -> [12] -> [4 3] all keep the same flat row-major order"
    (let [xs (vec (range 1 13))
          a (arr/from-vec backend xs [2 6])]
      (is (= (map double xs) (arr/->vec (t/reshape a [3 4]))))
      (is (= (map double xs) (arr/->vec (t/reshape a [12]))))
      (is (= (map double xs) (arr/->vec (t/reshape a [4 3]))))
      (is (= [4 3] (:shape (t/reshape a [4 3]))))))
  (testing "element-count mismatch throws"
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (t/reshape (arr/from-vec backend (range 12) [2 6]) [5 3])))))

;; --- squeeze / unsqueeze ---------------------------------------------------------

(deftest squeeze-unsqueeze-preserve-data
  (testing "squeeze removes all size-1 axes, data unchanged"
    (let [a (arr/from-vec backend (range 1 7) [1 2 1 3])
          sq (t/squeeze a)]
      (is (= [2 3] (:shape sq)))
      (is (= (map double (range 1 7)) (arr/->vec sq)))))
  (testing "squeeze a specific axis"
    (let [a (arr/from-vec backend (range 1 7) [2 1 3])]
      (is (= [2 3] (:shape (t/squeeze a 1))))
      (is (thrown? #?(:clj Exception :cljs js/Error) (t/squeeze a 0)))))
  (testing "unsqueeze inserts a size-1 axis, data unchanged"
    (let [a (arr/from-vec backend [1 2 3 4 5 6] [2 3])
          u0 (t/unsqueeze a 0)
          u2 (t/unsqueeze a 2)]
      (is (= [1 2 3] (:shape u0)))
      (is (= [2 3 1] (:shape u2)))
      (is (= (arr/->vec a) (arr/->vec u0) (arr/->vec u2)))))
  (testing "squeeze then unsqueeze round-trips shape and data"
    (let [a (arr/from-vec backend (range 1 7) [1 2 3])
          rt (t/unsqueeze (t/squeeze a 0) 0)]
      (is (= (:shape a) (:shape rt)))
      (is (= (arr/->vec a) (arr/->vec rt))))))

;; --- transpose --------------------------------------------------------------------

(deftest transpose-2d-matches-hand-computed
  (testing "default (full-reverse) transpose of a 2x3 matrix"
    ;; [[1 2 3]
    ;;  [4 5 6]]  ->  [[1 4]
    ;;                 [2 5]
    ;;                 [3 6]]
    (let [a (arr/from-vec backend [1 2 3 4 5 6] [2 3])
          at (t/transpose a)]
      (is (= [3 2] (:shape at)))
      (is (= [1.0 4.0 2.0 5.0 3.0 6.0] (arr/->vec at)))))
  (testing "double transpose is the identity"
    (let [a (arr/from-vec backend (range 1 13) [3 4])]
      (is (= (arr/->vec a) (arr/->vec (t/transpose (t/transpose a)))))
      (is (= (:shape a) (:shape (t/transpose (t/transpose a))))))))

(deftest transpose-3d-explicit-perm-matches-naive
  (testing "perm [2 0 1] on a 2x3x4 tensor matches an independently-written naive permute"
    (let [shape [2 3 4]
          xs (vec (range (reduce * shape)))
          a (arr/from-vec backend xs shape)
          perm [2 0 1]
          out-shape (mapv shape perm)
          ;; naive reference: out[i2][i0][i1] = in[i0][i1][i2], written from scratch
          ;; (not reusing num.tensor's stride math)
          in-at (fn [i0 i1 i2] (nth xs (+ (* i0 3 4) (* i1 4) i2)))
          naive (vec (for [i2 (range 4) i0 (range 2) i1 (range 3)]
                       (double (in-at i0 i1 i2))))
          out (t/transpose a perm)]
      (is (= out-shape (:shape out)))
      (is (= naive (arr/->vec out))))))

(deftest transpose-validates-perm
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (t/transpose (arr/from-vec backend (range 6) [2 3]) [0])))
  (is (thrown? #?(:clj Exception :cljs js/Error)
               (t/transpose (arr/from-vec backend (range 6) [2 3]) [0 0]))))

;; --- axis reductions ------------------------------------------------------------

(def r2x3 [1 2 3 4 5 6]) ; [[1 2 3] [4 5 6]]

(deftest sum-axis-2d-full
  (let [a (arr/from-vec backend r2x3 [2 3])]
    (testing "axis 0 (down columns): naive col sums [1+4 2+5 3+6]"
      (is (= [5.0 7.0 9.0] (arr/->vec (t/sum a 0))))
      (is (= [3] (:shape (t/sum a 0))))
      (is (= [1 3] (:shape (t/sum a 0 {:keepdims? true})))))
    (testing "axis 1 (across rows): naive row sums [1+2+3 4+5+6]"
      (is (= [6.0 15.0] (arr/->vec (t/sum a 1))))
      (is (= [2] (:shape (t/sum a 1))))
      (is (= [2 1] (:shape (t/sum a 1 {:keepdims? true})))))
    (testing "no axis = full reduction, same total as num.core/sum"
      (is (= (nm/sum a) (arr/->scalar (t/sum a))))
      (is (= [] (:shape (t/sum a)))))))

(deftest amax-amin-mean-axis-2d
  (let [a (arr/from-vec backend r2x3 [2 3])]
    (testing "amax/amin along axis 0"
      (is (= [4.0 5.0 6.0] (arr/->vec (t/amax a 0))))
      (is (= [1.0 2.0 3.0] (arr/->vec (t/amin a 0)))))
    (testing "amax/amin along axis 1"
      (is (= [3.0 6.0] (arr/->vec (t/amax a 1))))
      (is (= [1.0 4.0] (arr/->vec (t/amin a 1)))))
    (testing "mean along axis 0 and axis 1, hand-computed"
      (is (= [2.5 3.5 4.5] (arr/->vec (t/mean a 0))))
      (is (= [2.0 5.0] (arr/->vec (t/mean a 1))))
      (is (= (/ (reduce + r2x3) 6.0) (arr/->scalar (t/mean a)))))))

(deftest reduce-multi-axis-3d-matches-naive
  (testing "sum over axes [0 2] of a 2x3x4 tensor, matches an independently-written naive loop"
    (let [shape [2 3 4]
          xs (vec (range 1 (inc (reduce * shape))))
          a (arr/from-vec backend xs shape)
          at (fn [i j k] (nth xs (+ (* i 3 4) (* j 4) k)))
          naive (vec (for [j (range 3)]
                       (double (reduce + (for [i (range 2) k (range 4)] (at i j k))))))
          out (t/sum a [0 2])]
      (is (= [3] (:shape out)))
      (is (= naive (arr/->vec out)))
      (is (= [1 3 1] (:shape (t/sum a [0 2] {:keepdims? true})))))))

;; --- batched matmul ---------------------------------------------------------------

(deftest matmul-2d-matches-num-core
  (testing "plain 2x2 matmul agrees with num.core/matmul"
    (let [A (arr/from-vec backend [1 2 3 4] [2 2])
          B (arr/from-vec backend [5 6 7 8] [2 2])]
      (is (= (arr/->vec (nm/matmul A B)) (arr/->vec (t/matmul A B))))
      (is (= [2 2] (:shape (t/matmul A B)))))))

(deftest matmul-batched-matches-naive
  (testing "2 batches of 2x3 · 3x2, each batch checked against an independent naive product"
    (let [A (arr/from-vec backend (range 1 13) [2 2 3])   ; batch of 2, each 2x3
          B (arr/from-vec backend (range 1 13) [2 3 2])   ; batch of 2, each 3x2
          av (vec (range 1 13)) bv (vec (range 1 13))
          a-at (fn [b i k] (nth av (+ (* b 6) (* i 3) k)))
          b-at (fn [b k j] (nth bv (+ (* b 6) (* k 2) j)))
          naive (vec (for [b (range 2) i (range 2) j (range 2)]
                       (double (reduce + (for [k (range 3)] (* (a-at b i k) (b-at b k j)))))))
          out (t/matmul A B)]
      (is (= [2 2 2] (:shape out)))
      (is (= naive (arr/->vec out))))))

(deftest matmul-batch-broadcast
  (testing "a single 2x3 matrix batch-broadcasts against 4 batches of 3x2"
    (let [A (arr/from-vec backend [1 2 3 4 5 6] [1 2 3])         ; 1 batch of 2x3
          Bv (vec (range 1 25))
          B (arr/from-vec backend Bv [4 3 2])                    ; 4 batches of 3x2
          av [1 2 3 4 5 6]
          a-at (fn [i k] (nth av (+ (* i 3) k)))
          b-at (fn [b k j] (nth Bv (+ (* b 6) (* k 2) j)))
          naive (vec (for [b (range 4) i (range 2) j (range 2)]
                       (double (reduce + (for [k (range 3)] (* (a-at i k) (b-at b k j)))))))
          out (t/matmul A B)]
      (is (= [4 2 2] (:shape out)))
      (is (= naive (arr/->vec out))))))

;; --- rank-4 coverage (ADR-2607051400 §Phase 1: "arbitrary rank ... at least 4-D") ----
;; The shape math above (`row-major-strides`/`unravel`/`ravel`/`broadcast-shapes`) is
;; rank-generic (plain `(count shape)` loops, no rank ceiling anywhere), but genericity
;; claimed and genericity TESTED are different things — this suite exercises reshape,
;; transpose, axis-reduction, and batched matmul explicitly at rank 4, each against an
;; independently-written naive reference, so 4-D is verified, not just asserted.

(deftest rank-4-reshape-and-transpose
  (testing "reshape a 2x2x3x2 (24 elems) tensor to 4x6, data order unchanged"
    (let [xs (vec (range 1 25))
          a (arr/from-vec backend xs [2 2 3 2])]
      (is (= (map double xs) (arr/->vec (t/reshape a [4 6]))))
      (is (= [4 6] (:shape (t/reshape a [4 6]))))))
  (testing "transpose perm [3 1 0 2] on a 2x3x2x4 tensor matches an independent naive permute"
    (let [shape [2 3 2 4]
          xs (vec (range (reduce * shape)))
          a (arr/from-vec backend xs shape)
          perm [3 1 0 2]
          out-shape (mapv shape perm)
          at (fn [i0 i1 i2 i3] (nth xs (+ (* i0 3 2 4) (* i1 2 4) (* i2 4) i3)))
          ;; out[i3][i1][i0][i2] = in[i0][i1][i2][i3], written from scratch
          naive (vec (for [i3 (range 4) i1 (range 3) i0 (range 2) i2 (range 2)]
                       (double (at i0 i1 i2 i3))))
          out (t/transpose a perm)]
      (is (= out-shape (:shape out)))
      (is (= naive (arr/->vec out))))))

(deftest rank-4-axis-reduction-matches-naive
  (testing "sum over axes [1 3] of a 2x3x2x4 tensor matches an independent naive loop"
    (let [shape [2 3 2 4]
          xs (vec (range 1 (inc (reduce * shape))))
          a (arr/from-vec backend xs shape)
          at (fn [i0 i1 i2 i3] (nth xs (+ (* i0 3 2 4) (* i1 2 4) (* i2 4) i3)))
          naive (vec (for [i0 (range 2) i2 (range 2)]
                       (double (reduce + (for [i1 (range 3) i3 (range 4)] (at i0 i1 i2 i3))))))
          out (t/sum a [1 3])]
      (is (= [2 2] (:shape out)))
      (is (= naive (arr/->vec out)))
      (is (= [2 1 2 1] (:shape (t/sum a [1 3] {:keepdims? true})))))))

(deftest rank-4-batched-matmul-matches-naive
  (testing "matmul with a 2x3 batch grid (rank-4 operands) matches an independent naive product"
    (let [A (arr/from-vec backend (range 1 (inc (* 2 3 4 5))) [2 3 4 5])   ; 2x3 batches of 4x5
          B (arr/from-vec backend (range 1 (inc (* 2 3 5 6))) [2 3 5 6])  ; 2x3 batches of 5x6
          av (vec (range 1 (inc (* 2 3 4 5)))) bv (vec (range 1 (inc (* 2 3 5 6))))
          a-at (fn [b0 b1 i k] (nth av (+ (* b0 3 4 5) (* b1 4 5) (* i 5) k)))
          b-at (fn [b0 b1 k j] (nth bv (+ (* b0 3 5 6) (* b1 5 6) (* k 6) j)))
          naive (vec (for [b0 (range 2) b1 (range 3) i (range 4) j (range 6)]
                       (double (reduce + (for [k (range 5)] (* (a-at b0 b1 i k) (b-at b0 b1 k j)))))))
          out (t/matmul A B)]
      (is (= [2 3 4 6] (:shape out)))
      (is (= naive (arr/->vec out))))))
