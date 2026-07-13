(ns num.autograd
  "Minimal reverse-mode automatic differentiation (ADR-2607131500 Phase 1,
  'training'). Deliberately NOT a general training framework — the honest
  minimal form of 'training capability': real, correct gradients (verified
  against numerical/finite-difference gradients, the standard autograd
  correctness check — see test/num/autograd_test.cljc) for exactly the ops a
  tiny linear/relu/linear/softmax(+mse-loss) network needs, matching
  torch-clj's own README example architecture. CPU-backend only (`num.cpu`)
  — num.tensor's sync-only host-round-trip design (see its own docstring)
  means this doesn't reach the async Deno GPU backend either, same
  documented gap as the rest of the tensor layer.

  Design: a small tape-based (Wengert-list) autograd. `with-tape` binds a
  fresh tape; every op below (`matmul*`/`add-bias*`/`relu*`/`softmax*`/
  `mse-loss*`) builds its forward result AND records itself + a
  `backward-fn` closure onto the tape. `backward!` seeds the output's
  gradient and replays the tape in REVERSE (creation) order, each node's
  backward-fn reading its own accumulated upstream gradient and pushing
  gradient contributions into its parents' `:grad` atoms. This is correct
  for the acyclic, single-path sequential graphs a `torch.model/sequential`
  produces — it does NOT handle branching/shared-subgraph reuse (a value
  used by two different downstream ops) or control flow, which a general
  autograd needs and this one deliberately does not attempt."
  (:require [num.array :as arr]
            [num.core :as nm]
            [num.tensor :as t]))

(def ^:dynamic *tape* nil)

(defrecord Value [data grad backward-fn parents])

(defn- accumulate! [v contrib]
  (swap! (:grad v) (fn [g] (if g (t/add g contrib) contrib))))

(defn- record! [v]
  (when *tape* (swap! *tape* conj v))
  v)

(defn- node [data parents backward-fn]
  (record! (->Value data (atom nil) backward-fn parents)))

(defn value
  "Wrap `data` (an NDArray) as a graph leaf — a parameter (its `:grad` is
  read after `backward!`) or an input (its `:grad` is simply unused)."
  [data]
  (node data [] (fn [_])))

(defmacro with-tape
  "Run `body` with a fresh tape bound. Returns `[result tape-atom]`."
  [& body]
  `(binding [*tape* (atom [])]
     [(do ~@body) *tape*]))

(defn backward!
  "Seed `v`'s gradient with `seed` (an NDArray matching `v`'s shape — for a
  scalar loss node, a 1-element NDArray) and propagate through `tape` (the
  second element `with-tape` returned) in reverse creation order."
  [v seed tape]
  (accumulate! v seed)
  (doseq [nd (reverse @tape)]
    ((:backward-fn nd) nd)))

;; --- ops -------------------------------------------------------------------

(defn matmul*
  "z = x @ W. dL/dx = dL/dz @ W^T; dL/dW = x^T @ dL/dz."
  [x W]
  (node (nm/matmul (:data x) (:data W))
        [x W]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (nm/matmul g (t/transpose (:data W))))
            (accumulate! W (nm/matmul (t/transpose (:data x)) g))))))

(defn add-bias*
  "z = x + b, b broadcast over x's leading (batch) axis. dL/dx = dL/dz;
  dL/db = sum of dL/dz over the batch axis (every batch row's gradient
  contributes to the one shared bias vector)."
  [x b]
  (node (t/add (:data x) (:data b))
        [x b]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x g)
            (accumulate! b (t/sum g 0))))))

(defn- relu-grad
  "Elementwise derivative of relu at `x-data`: 1.0 where x>0, else 0.0 (the
  standard, if technically-debatable-at-exactly-0, convention). A small host
  round-trip — the same documented tradeoff every num.tensor op already
  makes, not a new one."
  [x-data]
  (arr/from-vec (:backend x-data)
                (mapv #(if (pos? %) 1.0 0.0) (arr/->vec x-data))
                (:shape x-data)))

(defn relu*
  [x]
  (node (nm/relu (:data x))
        [x]
        (fn [self]
          (when-let [g @(:grad self)]
            (accumulate! x (nm/mul g (relu-grad (:data x))))))))

(defn softmax*
  "y = softmax(x) along the last axis. Standard softmax-Jacobian-vector
  product: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j)) (per row)."
  [x]
  (let [y (t/softmax (:data x))]
    (node y
          [x]
          (fn [self]
            (when-let [g @(:grad self)]
              (let [gy (nm/mul g y)
                    s (t/sum gy (dec (count (:shape y))) {:keepdims? true})
                    dx (nm/mul y (t/sub g s))]
                (accumulate! x dx)))))))

(defn mse-loss*
  "L = mean((pred - target)^2) over every element, a scalar (shape `[]`).
  `target` is a plain NDArray, not a Value — nothing needs its gradient.
  dL/dpred = 2*(pred - target) / N."
  [pred target]
  (let [diff (t/sub (:data pred) target)
        n (double (arr/nelems (:shape diff)))
        loss-val (/ (nm/sum (nm/mul diff diff)) n)]
    (node (arr/from-vec (:backend (:data pred)) [loss-val] [])
          [pred]
          (fn [self]
            (when-let [g @(:grad self)]
              (let [scale (* 2.0 (arr/->scalar g) (/ 1.0 n))
                    diff-copy (arr/from-vec (:backend diff) (arr/->vec diff) (:shape diff))]
                (accumulate! pred (nm/scal! scale diff-copy))))))))
