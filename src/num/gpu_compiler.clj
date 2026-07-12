(ns num.gpu-compiler
  "Build-time/runtime bridge from num operators to kotoba compiler accelerator
  KIR. Dependency direction is num -> compiler; compiler never imports num."
  (:require [kotoba.compiler.accelerator :as accelerator]))

(def kernel-specs
  (into {}
        (concat
         (for [operator [:add :sub :mul :div]]
           [[:ewise operator] {:name (str "num_ewise_" (name operator) "_f32")
                               :op :ewise :operator operator :workgroup-size 256}])
         (for [operator [:sum :max :min]]
           [[:reduce operator] {:name (str "num_reduce_" (name operator) "_f32")
                                :op :reduce :operator operator :workgroup-size 256}]))))

(defn kernel-kir [kind operator]
  (let [{:keys [name op operator workgroup-size] :as spec} (get kernel-specs [kind operator])]
    (when-not spec (throw (ex-info "unsupported compiled numerical kernel" {:kind kind :operator operator})))
    (accelerator/kernel name op operator {:workgroup-size workgroup-size})))

(defn compile-kernel [kind operator target]
  (-> (accelerator/compile-kernel (kernel-kir kind operator) target)
      accelerator/verify-artifact!))

(defn artifact-registry []
  (into (sorted-map)
        (for [[[kind operator] _] kernel-specs target [:wgsl-v1 :cuda-v1]
              :let [compiled (compile-kernel kind operator target)]]
          [[kind operator target]
           (select-keys compiled [:format :target :kir-sha256 :code-sha256 :limits :code])])))
