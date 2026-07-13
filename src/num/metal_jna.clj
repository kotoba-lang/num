(ns num.metal-jna
  "JNA adapter for the opt-in native Metal runtime."
  (:require [num.metal :as metal])
  (:import [com.sun.jna Function Memory NativeLibrary Pointer]
           [com.sun.jna.ptr FloatByReference PointerByReference]))

(defn- function [^NativeLibrary library name] (.getFunction library name))
(defn- invoke-int [^Function f args] (.invokeInt f (object-array args)))
(defn- checked [driver operation status data]
  (when-not (zero? status)
    (let [library (:library driver) context (:context driver)
          message (.invokeString (function library "num_metal_last_error")
                                 (object-array [context]) false)]
      (throw (ex-info (str "Metal " operation " failed")
                      (assoc data :status status :native-error message))))))

(defrecord JnaMetalDriver [^NativeLibrary library ^Pointer context info kernels]
  java.io.Closeable
  (close [this]
    (doseq [kernel @kernels]
      (checked this "kernel destroy"
               (invoke-int (function library "num_metal_kernel_destroy") [context kernel]) {}))
    (reset! kernels [])
    (checked this "context destroy"
             (invoke-int (function library "num_metal_destroy") [context]) {})
    (.close library))
  metal/IMetalDriver
  (-metal-info [_] info)
  (-metal-alloc-f32 [this n]
    (let [out (PointerByReference.) s (invoke-int (function library "num_metal_malloc_f32") [context (long n) out])]
      (checked this "allocation" s {:n n}) (.getValue out)))
  (-metal-free [this ptr]
    (checked this "free" (invoke-int (function library "num_metal_free") [context ptr]) {}))
  (-metal-h2d [this ptr xs]
    (let [values (float-array (map float xs))]
      (checked this "host-to-device copy"
               (invoke-int (function library "num_metal_h2d_f32") [context ptr values (long (count values))]) {})))
  (-metal-d2h [this ptr n]
    (let [values (float-array n)]
      (checked this "device-to-host copy"
               (invoke-int (function library "num_metal_d2h_f32") [context values ptr (long n)]) {})
      (vec values)))
  (-metal-axpy! [this a x y n] (checked this "AXPY" (invoke-int (function library "num_metal_axpy") [context (float a) x y (int n)]) {}))
  (-metal-scal! [this a x n] (checked this "SCAL" (invoke-int (function library "num_metal_scal") [context (float a) x (int n)]) {}))
  (-metal-dot [this x y n] (let [out (FloatByReference.) s (invoke-int (function library "num_metal_dot") [context x y (int n) out])]
                             (checked this "DOT" s {}) (.getValue out)))
  (-metal-nrm2 [this x n] (let [out (FloatByReference.) s (invoke-int (function library "num_metal_nrm2") [context x (int n) out])]
                            (checked this "NRM2" s {}) (.getValue out)))
  (-metal-ewise! [this op x y z n]
    (checked this "elementwise" (invoke-int (function library "num_metal_ewise")
                                             [context ({:add 0 :sub 1 :mul 2 :div 3} op) x y z (int n)]) {:op op}))
  (-metal-reduce [this op x n]
    (let [out (FloatByReference.) s (invoke-int (function library "num_metal_reduce")
                                                [context ({:sum 0 :max 1 :min 2} op) x (int n) out])]
      (checked this "reduction" s {:op op}) (.getValue out)))
  (-mps-gemv! [this a A m n x b y]
    (checked this "MPS GEMV" (invoke-int (function library "num_metal_mps_gemv")
                                     [context (float a) A (int m) (int n) x (float b) y]) {}))
  (-mps-gemm! [this a A m k B n b C]
    (checked this "MPS GEMM" (invoke-int (function library "num_metal_mps_gemm")
                                     [context (float a) A (int m) (int k) B (int n) (float b) C]) {}))
  (-metal-spmv! [this csr x y]
    (let [rp (int-array (:row-ptr csr)) ci (int-array (:col-idx csr)) values (float-array (map float (:vals csr)))]
      (checked this "CSR SpMV" (invoke-int (function library "num_metal_spmv")
                                           [context (int (:n-rows csr)) (int (:n-cols csr)) (int (:nnz csr)) rp ci values x y]) {})))
  metal/ICompiledMetalDriver
  (-compile-msl [this artifact]
    (let [out (PointerByReference.) name (get-in artifact [:kir :kernel/name])
          s (invoke-int (function library "num_metal_compile") [context (:code artifact) name out])]
      (checked this "runtime MSL compilation" s {:kernel name :code-sha256 (:code-sha256 artifact)})
      (let [kernel (.getValue out)] (swap! kernels conj kernel) kernel)))
  (-compiled-metal-ewise! [this kernel x y z n workgroup-size]
    (checked this "compiled elementwise launch" (invoke-int (function library "num_metal_launch_ewise")
                                                             [context kernel x y z (int n) (int workgroup-size)]) {}))
  (-compiled-metal-reduce [this kernel op x n workgroup-size]
    (let [out (FloatByReference.) s (invoke-int (function library "num_metal_launch_reduce")
                                                [context kernel ({:sum 0 :max 1 :min 2} op) x (int n) (int workgroup-size) out])]
      (checked this "compiled reduction launch" s {:op op}) (.getValue out))))

(defn jna-driver
  ([] (jna-driver "num_metal"))
  ([library-name]
   (let [library (NativeLibrary/getInstance library-name) out (PointerByReference.)
         status (invoke-int (function library "num_metal_create") [out])]
     (when-not (zero? status) (.close library) (throw (ex-info "Metal context creation failed" {:status status})))
     (let [context (.getValue out) name-memory (Memory. 256)
           name-status (invoke-int (function library "num_metal_device_name") [context name-memory (long 256)])]
       (when-not (zero? name-status)
         (invoke-int (function library "num_metal_destroy") [context]) (.close library)
         (throw (ex-info "Metal device query failed" {:status name-status})))
       (->JnaMetalDriver library context
                         {:metal/device (.getString name-memory 0)
                          :metal/family "runtime-metal"
                          :metal/os-version (System/getProperty "os.version")
                          :metal/compiler-version "MTLLibrary runtime MSL"
                          :metal/native-api "Metal"
                          :metal/dense-provider "MetalPerformanceShaders"}
                         (atom []))))))

(defn backend []
  (let [driver (jna-driver)] {:driver driver :backend (metal/metal-backend driver)}))
