(ns dl4clj.loss-functions
  (:require [clojure.string :as cs])
  (:import (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)))

::negativeloglikelihood

(defmacro kw->enum [pre k]
  (-> k name cs/upper-case (cs/replace #"-" "_") (->> (str pre)) symbol))

(defmacro loss [kw] `(kw->enum "LossFunctions$LossFunction/" ~kw))

;{"dl4clj.loss-functions" "org.nd4j.linalg.lossfunctions"}
;
;((juxt namespace name)
;  ::negativeloglikelihood)
;
;(kw->enum "dl4clj.loss-functions/" :negativeloglikelihood)