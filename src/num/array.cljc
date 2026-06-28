(ns num.array
  "NDArray — the value users hold. It is just `{:backend :handle :shape}`: a shape
  plus an opaque device buffer on some backend. Because it is a value, a whole
  computation is checkpointable/serializable at the host boundary (copy to a
  vector), matching the audit-ledger ethos of the rest of com-junkawasaki.

  Construct with `from-vec` / `zeros` / `from-fn`; read back with `->vec` /
  `->scalar`. Ops live in `num.core`."
  (:require [num.protocol :as p]))

(defrecord NDArray [backend handle shape])

(defn nelems
  "Element count of a shape vector."
  [shape]
  (reduce * 1 shape))

(defn from-vec
  "Upload host data `xs` (row-major) as an NDArray of `shape` on `backend`."
  [backend xs shape]
  (->NDArray backend (p/-copy-from-host backend xs) shape))

(defn zeros
  "An NDArray of `shape` on `backend` (backend buffers are zero-initialized for the
  CPU reference; GPU backends clear on alloc)."
  [backend shape]
  (->NDArray backend (p/-alloc backend (nelems shape)) shape))

(defn from-fn
  "Build a `shape` NDArray on `backend` from `(f linear-index)`."
  [backend shape f]
  (from-vec backend (mapv f (range (nelems shape))) shape))

(defn ->vec
  "Copy an NDArray back to a host vector of doubles (row-major)."
  [a]
  (p/-copy-to-host (:backend a) (:handle a) (nelems (:shape a))))

(defn ->scalar
  "Read a 1-element NDArray as a host double."
  [a]
  (first (->vec a)))

(defn like
  "An uninitialized NDArray with the same backend/shape as `a`."
  [a]
  (zeros (:backend a) (:shape a)))
