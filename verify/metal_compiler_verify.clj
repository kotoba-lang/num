(ns metal-compiler-verify
  (:require [num.gpu-compiler :as gpu]))

(defn -main [& _]
  (let [ewise (gpu/compile-kernel :ewise :add :msl-v1)
        reduction (gpu/compile-kernel :reduce :sum :msl-v1)
        process (ProcessBuilder. ["xcrun" "swift" "verify/metal_compiler_runtime.swift"])
        env (.environment process)]
    (.put env "NUM_MSL_EWISE" (:code ewise))
    (.put env "NUM_MSL_REDUCE" (:code reduction))
    (.inheritIO process)
    (let [exit (.waitFor (.start process))]
      (when-not (zero? exit) (throw (ex-info "Metal compiler runtime verification failed" {:exit exit})))
      (println "KIR" (:kir-sha256 ewise) "MSL" (:code-sha256 ewise)))))
