(ns num.quantized
  "Packed GGML inference matrices that remain quantized through matmul."
  (:require [num.array :as arr]
            [num.protocol :as p]))

(defrecord QuantizedMatrix [backend handle shape quant-type byte-count block-size])

(defn matrix
  "Upload GGML row-quantized bytes. `source-shape` is GGML's logical `[out,in]`;
  the returned matrix presents `[in,out]` so it composes with `num.core/matmul`."
  [backend bytes source-shape quant-type]
  (let [[out in] source-shape
        block-size ({:q4-k 256 :q6-k 256 :q8-0 32} quant-type)
        bytes-per-block ({:q4-k 144 :q6-k 210 :q8-0 34} quant-type)]
    (when-not (and (= 2 (count source-shape)) block-size bytes-per-block
                   (pos-int? out) (pos-int? in)
                   (zero? (mod in block-size))
                   (= (count bytes) (* out (quot in block-size) bytes-per-block))
                   (satisfies? p/IQuantizedOps backend))
      (throw (ex-info "invalid or unsupported packed quantized matrix"
                      {:source-shape source-shape :quant-type quant-type
                       :bytes (count bytes)})))
    (->QuantizedMatrix backend
                       (p/-quantized-from-host backend bytes
                                                {:quant-type quant-type
                                                 :rows out :cols in})
                       [in out] quant-type (count bytes) block-size)))

(defn matrix? [value] (instance? QuantizedMatrix value))

(defn matmul
  "Multiply dense f32 `input` by packed `weight`, returning dense f32 output."
  [input weight]
  (let [[m k] (:shape input) [wk n] (:shape weight)]
    (when-not (and (matrix? weight) (= k wk)
                   (= (:backend input) (:backend weight))
                   (= :f32 (:dtype input :f32)))
      (throw (ex-info "quantized matmul dimensions/backend are incompatible"
                      {:input (:shape input) :weight (:shape weight)})))
    (assoc (arr/->NDArray
            (:backend input)
            (p/-quantized-matmul (:backend input) (:handle input) (:handle weight)
                                 {:quant-type (:quant-type weight)
                                  :m m :k k :n n
                                  :blocks-per-row (quot k (:block-size weight))})
            [m n])
           :dtype :f32)))

(defn release! [weight]
  (p/-free (:backend weight) (:handle weight))
  nil)
