(ns num.cuda
  "CUDA IBackend. Native CUDA ownership is injected through ICudaDriver so
  num's portable core never loads a CUDA library on non-NVIDIA hosts.

  Driver implementations map level-1/2/3 calls to cuBLAS and CSR SpMV to
  cuSPARSE. Dense matrices are row-major at this boundary; a cuBLAS adapter is
  responsible for the conventional operand/transpose swap required by its
  column-major API."
  (:require [num.protocol :as p]))

(defprotocol ICudaDriver
  (-device-info [driver] "Map containing device/runtime/library versions.")
  (-malloc-f32 [driver n] "Allocate n float32 values and return an opaque pointer.")
  (-free-ptr [driver ptr])
  (-h2d-f32 [driver ptr xs])
  (-d2h-f32 [driver ptr n])
  (-cuda-axpy! [driver alpha x y n])
  (-cuda-scal! [driver alpha x n])
  (-cuda-dot [driver x y n])
  (-cuda-nrm2 [driver x n])
  (-cuda-ewise! [driver op x y z n])
  (-cuda-reduce [driver op x n])
  (-cublas-gemv! [driver alpha A m n x beta y])
  (-cublas-gemm! [driver alpha A m k B n beta C])
  (-cusparse-spmv! [driver csr x y]))

(defrecord CudaHandle [owner ptr n released?])

(defn- checked-handle [backend handle required]
  (when-not (and (instance? CudaHandle handle)
                 (identical? (:owner handle) backend)
                 (not @(:released? handle))
                 (<= required (:n handle)))
    (throw (ex-info "invalid, foreign, undersized or released CUDA handle"
                    {:required required :handle-size (:n handle)
                     :released? (some-> handle :released? deref)})))
  (:ptr handle))

(declare ->CudaBackend)

(deftype CudaBackend [driver device-info]
  p/IBackend
  (-backend-name [_] :cuda)
  (-alloc [this n]
    (when-not (pos-int? n) (throw (ex-info "CUDA allocation size must be positive" {:n n})))
    (->CudaHandle this (-malloc-f32 driver n) n (atom false)))
  (-free [this handle]
    (checked-handle this handle 0)
    (-free-ptr driver (:ptr handle))
    (reset! (:released? handle) true)
    nil)
  (-copy-from-host [this xs]
    (let [values (mapv float xs) handle (p/-alloc this (count values))]
      (-h2d-f32 driver (:ptr handle) values) handle))
  (-copy-to-host [this handle n]
    (mapv double (-d2h-f32 driver (checked-handle this handle n) n)))
  (-axpy [this alpha x y n]
    (-cuda-axpy! driver (float alpha) (checked-handle this x n) (checked-handle this y n) n) y)
  (-scal [this alpha x n]
    (-cuda-scal! driver (float alpha) (checked-handle this x n) n) x)
  (-dot [this x y n]
    (double (-cuda-dot driver (checked-handle this x n) (checked-handle this y n) n)))
  (-nrm2 [this x n] (double (-cuda-nrm2 driver (checked-handle this x n) n)))
  (-ewise [this op x y n]
    (when-not (#{:add :sub :mul :div} op) (throw (ex-info "unsupported CUDA ewise op" {:op op})))
    (let [z (p/-alloc this n)]
      (-cuda-ewise! driver op (checked-handle this x n) (checked-handle this y n) (:ptr z) n) z))
  (-reduce [this op x n]
    (when-not (#{:sum :max :min} op) (throw (ex-info "unsupported CUDA reduction" {:op op})))
    (double (-cuda-reduce driver op (checked-handle this x n) n)))
  (-gemv [this alpha A m n x beta y]
    (-cublas-gemv! driver (float alpha) (checked-handle this A (* m n)) m n
                   (checked-handle this x n) (float beta) (checked-handle this y m)) y)
  (-gemm [this alpha A m k B n beta C]
    (-cublas-gemm! driver (float alpha) (checked-handle this A (* m k)) m k
                   (checked-handle this B (* k n)) n (float beta) (checked-handle this C (* m n))) C)
  (-spmv [this csr x]
    (let [m (:n-rows csr) y (p/-alloc this m)]
      (checked-handle this x (:n-cols csr))
      (-cusparse-spmv! driver csr (:ptr x) (:ptr y)) y)))

(defn cuda-backend [driver]
  (when-not (satisfies? ICudaDriver driver)
    (throw (ex-info "CUDA backend requires an ICudaDriver" {})))
  (let [info (-device-info driver)]
    (when-not (and (map? info) (string? (:cuda/device info))
                   (string? (:cuda/runtime-version info))
                   (string? (:cuda/cublas-version info))
                   (string? (:cuda/cusparse-version info)))
      (throw (ex-info "CUDA driver did not report required provenance" {:device-info info})))
    (->CudaBackend driver info)))

(defn backend-info [backend]
  (when-not (instance? CudaBackend backend) (throw (ex-info "not a CUDA backend" {})))
  (.-device-info ^CudaBackend backend))
