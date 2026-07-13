(ns num.core
  "The public, backend-agnostic API. Every function reads its operands' backend
  and dispatches through `num.protocol/IBackend`, so the SAME program runs on the
  CPU reference, a WGSL/WebGPU pipeline, or cuBLAS — the caller never names a GPU.

  Shapes are row-major. Vectors are shape `[n]`, matrices `[m n]`. Level-1 ops
  (`axpy!`, `scal!`) mutate in place and return the target (BLAS convention);
  everything else returns a fresh NDArray."
  (:require [num.protocol :as p]
            [num.array :as arr]))

(defn- be [a] (:backend a))
(defn- n1 [a] (arr/nelems (:shape a)))

;; --- level-1 -----------------------------------------------------------------

(defn axpy!
  "y ← αx + y (in place); returns y."
  [alpha x y]
  (p/-axpy (be y) alpha (:handle x) (:handle y) (n1 y))
  y)

(defn scal!
  "x ← αx (in place); returns x."
  [alpha x]
  (p/-scal (be x) alpha (:handle x) (n1 x))
  x)

(defn dot
  "Σ xᵢyᵢ as a host double."
  [x y]
  (p/-dot (be x) (:handle x) (:handle y) (n1 x)))

(defn nrm2
  "Euclidean norm ‖x‖₂ as a host double."
  [x]
  (p/-nrm2 (be x) (:handle x) (n1 x)))

;; --- elementwise + reductions ------------------------------------------------

(defn- ewise [op x y]
  (arr/->NDArray (be x) (p/-ewise (be x) op (:handle x) (:handle y) (n1 x)) (:shape x)))

(defn add [x y] (ewise :add x y))
(defn sub [x y] (ewise :sub x y))
(defn mul [x y] (ewise :mul x y))         ; Hadamard (elementwise) product
(defn div [x y] (ewise :div x y))

(defn- ewise1 [op x]
  (arr/->NDArray (be x) (p/-ewise1 (be x) op (:handle x) (n1 x)) (:shape x)))

(defn exp [x] (ewise1 :exp x))
(defn relu [x] (ewise1 :relu x))
(defn neg [x] (ewise1 :neg x))

(defn sum [x] (p/-reduce (be x) :sum (:handle x) (n1 x)))
(defn amax [x] (p/-reduce (be x) :max (:handle x) (n1 x)))
(defn amin [x] (p/-reduce (be x) :min (:handle x) (n1 x)))

;; --- level-2 / level-3 -------------------------------------------------------

(defn matvec
  "y = A·x for dense A (shape [m n]) and vector x (shape [n]) → NDArray [m]."
  [A x]
  (let [b (be A) [m n] (:shape A) y (p/-alloc b m)]
    (p/-gemv b 1.0 (:handle A) m n (:handle x) 0.0 y)
    (arr/->NDArray b y [m])))

(defn matmul
  "C = A·B for dense A (shape [m k]) and B (shape [k n]) → NDArray [m n]."
  [A B]
  (let [b (be A) [m k] (:shape A) [_ n] (:shape B) C (p/-alloc b (* m n))]
    (p/-gemm b 1.0 (:handle A) m k (:handle B) n 0.0 C)
    (arr/->NDArray b C [m n])))

;; --- sparse ------------------------------------------------------------------

(defn spmv
  "Sparse mat-vec y = A·x for a CSR `csr` (num.sparse) and NDArray x → NDArray
  [n-rows]. The CSR is host structure; only x lives on the device."
  [backend csr x]
  (arr/->NDArray backend (p/-spmv backend csr (:handle x)) [(:n-rows csr)]))
