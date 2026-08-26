(ns num.array
  "NDArray — the value users hold. It is just `{:backend :handle :shape}`: a shape
  plus an opaque device buffer on some backend. Because it is a value, a whole
  computation is checkpointable/serializable at the host boundary (copy to a
  vector), matching the audit-ledger ethos of the rest of com-junkawasaki.

  Construct with `from-vec` / `zeros` / `from-fn`; read back with `->vec` /
  `->scalar`. Ops live in `num.core`."
  (:refer-clojure :exclude [cast])
  (:require [num.dtype :as dtype]
            [num.protocol :as p]))

(defrecord NDArray [backend handle shape])

(defn nelems
  "Element count of a shape vector."
  [shape]
  (reduce * 1 shape))

(defn from-vec
  "Upload host data `xs` (row-major) as an NDArray of `shape` on `backend`."
  ([backend xs shape] (from-vec backend xs shape :f32))
  ([backend xs shape dtype*]
   (dtype/check dtype*)
   (when-not (= (count xs) (nelems shape))
     (throw (ex-info "data length does not match shape"
                     {:count (count xs) :shape shape})))
   (if (= dtype* :f32)
     (assoc (->NDArray backend (p/-copy-from-host backend xs) shape) :dtype :f32)
     (do
       (when-not (satisfies? p/IDTypeStorage backend)
         (throw (ex-info "backend does not support typed storage"
                         {:backend (p/-backend-name backend) :dtype dtype*})))
       (assoc (->NDArray backend (p/-copy-from-host-dtype backend xs dtype*) shape)
              :dtype dtype*)))))

(defn zeros
  "An NDArray of `shape` on `backend` (backend buffers are zero-initialized for the
  CPU reference; GPU backends clear on alloc)."
  ([backend shape] (zeros backend shape :f32))
  ([backend shape dtype*]
   (dtype/check dtype*)
   (if (= dtype* :f32)
     (assoc (->NDArray backend (p/-alloc backend (nelems shape)) shape) :dtype :f32)
     (do
       (when-not (satisfies? p/IDTypeStorage backend)
         (throw (ex-info "backend does not support typed storage" {:dtype dtype*})))
       (assoc (->NDArray backend (p/-alloc-dtype backend (nelems shape) dtype*) shape)
              :dtype dtype*)))))

(defn from-fn
  "Build a `shape` NDArray on `backend` from `(f linear-index)`."
  [backend shape f]
  (from-vec backend (mapv f (range (nelems shape))) shape))

(defn ->vec
  "Copy an NDArray back to a host vector of doubles (row-major)."
  [a]
  (let [dtype* (or (:dtype a) :f32)]
    (if (= dtype* :f32)
      (p/-copy-to-host (:backend a) (:handle a) (nelems (:shape a)))
      (p/-copy-to-host-dtype (:backend a) (:handle a) (nelems (:shape a)) dtype*))))

(defn cast
  "Materialize `a` in physical `target-dtype` storage on the same backend."
  [a target-dtype]
  (let [source-dtype (or (:dtype a) :f32)
        backend (:backend a)]
    (cond
      (= source-dtype target-dtype) a
      (satisfies? p/ICastOps backend)
      (assoc (->NDArray backend
                        (p/-cast-dtype backend (:handle a) (nelems (:shape a))
                                       source-dtype target-dtype)
                        (:shape a))
             :dtype target-dtype)
      :else (from-vec backend (->vec a) (:shape a) target-dtype))))

(defn ->scalar
  "Read a 1-element NDArray as a host double."
  [a]
  (first (->vec a)))

(defn argmax-rows
  "Return one maximum-value column index per row without materializing logits
  on the host. The array must be contiguous f32 `[rows,cols]`, and its backend
  must implement `IDeviceSelection`. Async GPU backends return a Promise."
  [a]
  (let [[rows cols :as shape] (:shape a)
        backend (:backend a)]
    (when-not (and (= 2 (count shape)) (pos-int? rows) (pos-int? cols)
                   (= :f32 (or (:dtype a) :f32)))
      (throw (ex-info "argmax-rows requires a non-empty f32 matrix"
                      {:shape shape :dtype (:dtype a)})))
    (when-not (satisfies? p/IDeviceSelection backend)
      (throw (ex-info "backend does not support device row argmax"
                      {:backend (p/-backend-name backend)})))
    (p/-argmax-rows backend (:handle a) rows cols)))

(defn top-k-row
  "Return the best `k` `[token logit]` pairs for one row without materializing
  the logits matrix on the host. Repetition penalty is applied once per unique
  previous token. The device may mutate that row, so call this only for logits
  that are no longer needed by model compute. Async GPU backends return a
  Promise."
  ([a row k] (top-k-row a row k [] 1.0))
  ([a row k previous-tokens repetition-penalty]
   (let [[rows cols :as shape] (:shape a)
         backend (:backend a)
         previous-tokens (vec (distinct previous-tokens))]
     (when-not (and (= 2 (count shape)) (pos-int? rows) (pos-int? cols)
                    (= :f32 (or (:dtype a) :f32))
                    (int? row) (<= 0 row) (< row rows)
                    (pos-int? k) (<= k cols) (<= k 64)
                    (number? repetition-penalty) (<= 1.0 repetition-penalty)
                    (every? #(and (int? %) (<= 0 %) (< % cols)) previous-tokens))
       (throw (ex-info "top-k-row requires a valid f32 matrix row and k <= 64"
                       {:shape shape :dtype (:dtype a) :row row :k k
                        :repetition-penalty repetition-penalty})))
     (when-not (satisfies? p/IDeviceSelection backend)
       (throw (ex-info "backend does not support device row top-k"
                       {:backend (p/-backend-name backend)})))
     (p/-top-k-row backend (:handle a) rows cols row k previous-tokens
                   repetition-penalty))))

(defn like
  "An uninitialized NDArray with the same backend/shape as `a`."
  [a]
  (zeros (:backend a) (:shape a) (or (:dtype a) :f32)))

(defn release!
  "Explicitly release an NDArray's backing storage. Required for predictable
  GPU memory use in long-running workloads; CPU/GC backends may treat it as a
  no-op. The NDArray must not be used after this call."
  [a]
  (when a
    (p/-free (:backend a) (:handle a)))
  nil)

(defn release-all!
  "Release each distinct backing handle in `arrays` once. This is safe for
  reshape/transpose aliases that share a handle. All arrays become invalid."
  [arrays]
  (loop [remaining (seq arrays) seen #{}]
    (when-let [a (first remaining)]
      (let [handle (:handle a)]
        (if (contains? seen handle)
          (recur (next remaining) seen)
          (do (release! a)
              (recur (next remaining) (conj seen handle)))))))
  nil)
