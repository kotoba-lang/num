(ns num.cuda-native-verify
  (:require [num.contract :as contract] [num.cuda-jna :as jna]))

(defn -main [& _]
  (let [{:keys [driver backend]} (jna/backend)
        failures (atom [])]
    (try
      (println "CUDA:" (pr-str (:info driver)))
      (contract/verify backend
                       (fn [pass? label]
                         (println (if pass? "PASS" "FAIL") label)
                         (when-not pass? (swap! failures conj label))))
      (when (seq @failures)
        (throw (ex-info "native CUDA contract failed" {:operations @failures})))
      (println "CUDA native IBackend contract: 14/14 PASS")
      (finally (.close ^java.io.Closeable driver)))))
