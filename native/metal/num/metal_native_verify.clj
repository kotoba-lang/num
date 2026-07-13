(ns num.metal-native-verify
  (:require [num.contract :as contract] [num.metal-jna :as jna]))

(defn -main [& _]
  (let [{:keys [driver backend]} (jna/backend) failures (atom [])]
    (try
      (println "Metal:" (pr-str (:info driver)))
      (contract/verify backend (fn [pass? label]
                                 (println (if pass? "PASS" "FAIL") label)
                                 (when-not pass? (swap! failures conj label))))
      (when (seq @failures) (throw (ex-info "native Metal contract failed" {:operations @failures})))
      (println "Metal native IBackend contract: 14/14 PASS")
      (finally (.close ^java.io.Closeable driver)))))
