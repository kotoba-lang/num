(ns num.cuda-jna
  "JNA host adapter for native/cuda/libnum_cuda.so. Load only with
  `-M:cuda-jna`; CUDA/JNA remain absent from num's default dependency graph."
  (:require [num.cuda :as cuda])
  (:import [com.sun.jna Function Memory NativeLibrary Pointer]
           [com.sun.jna.ptr FloatByReference IntByReference PointerByReference]))

(defn- function [^NativeLibrary library name] (.getFunction library name))
(defn- invoke-int [^Function f args] (.invokeInt f (object-array args)))

(defrecord JnaCudaDriver [^NativeLibrary library ^Pointer context info]
  java.io.Closeable
  (close [_]
    (let [status (invoke-int (function library "num_cuda_destroy") [context])]
      (when-not (zero? status) (throw (ex-info "CUDA context destroy failed" {:status status}))))
    (.close library))
  cuda/ICudaDriver
  (-device-info [_] info)
  (-malloc-f32 [_ n]
    (let [out (PointerByReference.) status (invoke-int (function library "num_cuda_malloc_f32") [context (long n) out])]
      (when-not (zero? status) (throw (ex-info "CUDA malloc failed" {:status status :n n})))
      (.getValue out)))
  (-free-ptr [_ ptr]
    (let [status (invoke-int (function library "num_cuda_free") [context ptr])]
      (when-not (zero? status) (throw (ex-info "CUDA free failed" {:status status})))))
  (-h2d-f32 [_ ptr xs]
    (let [values (float-array (map float xs))
          status (invoke-int (function library "num_cuda_h2d_f32") [context ptr values (long (count values))])]
      (when-not (zero? status) (throw (ex-info "CUDA host-to-device copy failed" {:status status})))))
  (-d2h-f32 [_ ptr n]
    (let [values (float-array n)
          status (invoke-int (function library "num_cuda_d2h_f32") [context values ptr (long n)])]
      (when-not (zero? status) (throw (ex-info "CUDA device-to-host copy failed" {:status status})))
      (vec values)))
  (-cuda-axpy! [_ alpha x y n]
    (let [s (invoke-int (function library "num_cuda_axpy") [context (float alpha) x y (int n)])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SAXPY failed" {:status s})))))
  (-cuda-scal! [_ alpha x n]
    (let [s (invoke-int (function library "num_cuda_scal") [context (float alpha) x (int n)])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SSCAL failed" {:status s})))))
  (-cuda-dot [_ x y n]
    (let [out (FloatByReference.) s (invoke-int (function library "num_cuda_dot") [context x y (int n) out])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SDOT failed" {:status s}))) (.getValue out)))
  (-cuda-nrm2 [_ x n]
    (let [out (FloatByReference.) s (invoke-int (function library "num_cuda_nrm2") [context x (int n) out])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SNRM2 failed" {:status s}))) (.getValue out)))
  (-cuda-ewise! [_ op x y z n]
    (let [s (invoke-int (function library "num_cuda_ewise")
                        [context ({:add 0 :sub 1 :mul 2 :div 3} op) x y z (int n)])]
      (when-not (zero? s) (throw (ex-info "CUDA elementwise kernel failed" {:status s :op op})))))
  (-cuda-reduce [_ op x n]
    (let [out (FloatByReference.) s (invoke-int (function library "num_cuda_reduce")
                                                [context ({:sum 0 :max 1 :min 2} op) x (int n) out])]
      (when-not (zero? s) (throw (ex-info "CUDA reduction failed" {:status s :op op}))) (.getValue out)))
  (-cublas-gemv! [_ alpha A m n x beta y]
    (let [s (invoke-int (function library "num_cuda_gemv_row_major")
                        [context (float alpha) A (int m) (int n) x (float beta) y])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SGEMV failed" {:status s})))))
  (-cublas-gemm! [_ alpha A m k B n beta C]
    (let [s (invoke-int (function library "num_cuda_gemm_row_major")
                        [context (float alpha) A (int m) (int k) B (int n) (float beta) C])]
      (when-not (zero? s) (throw (ex-info "cuBLAS SGEMM failed" {:status s})))))
  (-cusparse-spmv! [_ csr x y]
    (let [rows (:n-rows csr) cols (:n-cols csr) nnz (:nnz csr)
          rp (int-array (seq (:row-ptr csr))) ci (int-array (seq (:col-idx csr)))
          values (float-array (map float (seq (:vals csr))))
          s (invoke-int (function library "num_cuda_spmv_csr")
                        [context (int rows) (int cols) (int nnz) rp ci values x y])]
      (when-not (zero? s) (throw (ex-info "cuSPARSE SpMV failed" {:status s})))))
  )

(defn jna-driver
  "Load libnum_cuda and create a driver for `device` (default 0). The returned
  driver is Closeable; close it after all CudaHandles have been freed."
  ([] (jna-driver "num_cuda" 0))
  ([library-name device]
   (let [library (NativeLibrary/getInstance library-name)
         out (PointerByReference.)
         status (invoke-int (function library "num_cuda_create") [(int device) out])]
     (when-not (zero? status)
       (.close library) (throw (ex-info "CUDA context creation failed" {:status status :device device})))
     (let [context (.getValue out) name-memory (Memory. 256)
           runtime (IntByReference.) driver-version (IntByReference.)
           cublas (IntByReference.) cusparse (IntByReference.)
           name-status (invoke-int (function library "num_cuda_device_name") [context name-memory (long 256)])
           version-status (invoke-int (function library "num_cuda_versions")
                                      [context runtime driver-version cublas cusparse])]
       (when-not (and (zero? name-status) (zero? version-status))
         (invoke-int (function library "num_cuda_destroy") [context]) (.close library)
         (throw (ex-info "CUDA provenance query failed" {:name-status name-status :version-status version-status})))
       (->JnaCudaDriver library context
                        {:cuda/device (.getString name-memory 0)
                         :cuda/runtime-version (str (.getValue runtime))
                         :cuda/driver-version (str (.getValue driver-version))
                         :cuda/cublas-version (str (.getValue cublas))
                         :cuda/cusparse-version (str (.getValue cusparse))})))))

(defn backend
  "Create {:driver Closeable :backend num.cuda/CudaBackend}."
  ([] (backend "num_cuda" 0))
  ([library-name device]
   (let [driver (jna-driver library-name device)]
     {:driver driver :backend (cuda/cuda-backend driver)})))
