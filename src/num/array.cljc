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
  (if (= (or (:dtype a) :f32) target-dtype)
    a
    (from-vec (:backend a) (->vec a) (:shape a) target-dtype)))

(defn ->scalar
  "Read a 1-element NDArray as a host double."
  [a]
  (first (->vec a)))

(defn like
  "An uninitialized NDArray with the same backend/shape as `a`."
  [a]
  (zeros (:backend a) (:shape a) (or (:dtype a) :f32)))
