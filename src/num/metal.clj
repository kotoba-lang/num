(ns num.metal
  "Native Metal/MPS IBackend behind an injected host driver."
  (:require [num.gpu-compiler :as gpu-compiler] [num.protocol :as p]))

(defprotocol IMetalDriver
  (-metal-info [d]) (-metal-alloc-f32 [d n]) (-metal-free [d ptr])
  (-metal-h2d [d ptr xs]) (-metal-d2h [d ptr n])
  (-metal-axpy! [d a x y n]) (-metal-scal! [d a x n])
  (-metal-dot [d x y n]) (-metal-nrm2 [d x n])
  (-metal-ewise! [d op x y z n]) (-metal-reduce [d op x n])
  (-mps-gemv! [d a A m n x b y]) (-mps-gemm! [d a A m k B n b C])
  (-metal-spmv! [d csr x y]))

(defprotocol ICompiledMetalDriver
  (-compile-msl [d artifact])
  (-compiled-metal-ewise! [d kernel x y z n workgroup-size])
  (-compiled-metal-reduce [d kernel op x n workgroup-size]))

(defrecord MetalHandle [owner ptr n released?])
(defn- ptr! [backend handle required]
  (when-not (and (instance? MetalHandle handle) (identical? backend (:owner handle))
                 (not @(:released? handle)) (<= required (:n handle)))
    (throw (ex-info "invalid, foreign, undersized or released Metal handle"
                    {:required required :released? (some-> handle :released? deref)})))
  (:ptr handle))

(deftype MetalBackend [driver info compiled]
  p/IBackend
  (-backend-name [_] :metal)
  (-alloc [this n] (when-not (pos-int? n) (throw (ex-info "Metal allocation size must be positive" {:n n})))
    (->MetalHandle this (-metal-alloc-f32 driver n) n (atom false)))
  (-free [this h] (-metal-free driver (ptr! this h 0)) (reset! (:released? h) true) nil)
  (-copy-from-host [this xs] (let [v (mapv float xs) h (p/-alloc this (count v))]
                               (-metal-h2d driver (:ptr h) v) h))
  (-copy-to-host [this h n] (mapv double (-metal-d2h driver (ptr! this h n) n)))
  (-axpy [this a x y n] (-metal-axpy! driver (float a) (ptr! this x n) (ptr! this y n) n) y)
  (-scal [this a x n] (-metal-scal! driver (float a) (ptr! this x n) n) x)
  (-dot [this x y n] (double (-metal-dot driver (ptr! this x n) (ptr! this y n) n)))
  (-nrm2 [this x n] (double (-metal-nrm2 driver (ptr! this x n) n)))
  (-ewise [this op x y n]
    (let [z (p/-alloc this n)]
      (if-let [kernel (get compiled [:ewise op])]
        (-compiled-metal-ewise! driver kernel (ptr! this x n) (ptr! this y n) (:ptr z) n 256)
        (-metal-ewise! driver op (ptr! this x n) (ptr! this y n) (:ptr z) n)) z))
  (-reduce [this op x n]
    (if-let [kernel (get compiled [:reduce op])]
      (double (-compiled-metal-reduce driver kernel op (ptr! this x n) n 256))
      (double (-metal-reduce driver op (ptr! this x n) n))))
  (-gemv [this a A m n x b y]
    (-mps-gemv! driver (float a) (ptr! this A (* m n)) m n (ptr! this x n) (float b) (ptr! this y m)) y)
  (-gemm [this a A m k B n b C]
    (-mps-gemm! driver (float a) (ptr! this A (* m k)) m k (ptr! this B (* k n)) n (float b) (ptr! this C (* m n))) C)
  (-spmv [this csr x] (let [y (p/-alloc this (:n-rows csr))]
                         (-metal-spmv! driver csr (ptr! this x (:n-cols csr)) (:ptr y)) y)))

(defn metal-backend [driver]
  (when-not (satisfies? IMetalDriver driver) (throw (ex-info "Metal backend requires IMetalDriver" {})))
  (let [info (-metal-info driver)]
    (when-not (and (map? info) (string? (:metal/device info)) (string? (:metal/os-version info))
                   (string? (:metal/family info)) (string? (:metal/compiler-version info)))
      (throw (ex-info "Metal driver did not report required provenance" {:metal-info info})))
    (let [artifacts (into (sorted-map)
                          (for [[[kind operator] _] gpu-compiler/kernel-specs
                                :let [a (gpu-compiler/compile-kernel kind operator :msl-v1)]]
                            [[kind operator] (select-keys a [:kir-sha256 :code-sha256 :limits])]))
          compiled (if (satisfies? ICompiledMetalDriver driver)
                     (into {} (for [[[kind operator] _] gpu-compiler/kernel-specs]
                                [[kind operator] (-compile-msl driver (gpu-compiler/compile-kernel kind operator :msl-v1))])) {})]
      (->MetalBackend driver (assoc info :metal/compiler-target :msl-v1
                                         :metal/compiler-mode (if (seq compiled) :runtime-msl :bootstrap)
                                         :metal/compiler-artifacts artifacts) compiled))))

(defn backend-info [backend]
  (when-not (instance? MetalBackend backend) (throw (ex-info "not a Metal backend" {})))
  (.-info ^MetalBackend backend))
