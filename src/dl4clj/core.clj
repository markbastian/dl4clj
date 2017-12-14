(ns dl4clj.core
  (:require [clojure.java.io :as io]
            [clojure.string :as cs]
            [dl4clj.loss-functions :as loss])
  (:import (org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder Updater)
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)
           (org.datavec.api.split FileSplit)
           (org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator)
           (org.datavec.api.records.reader.impl.csv CSVRecordReader)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.deeplearning4j.eval Evaluation)
           (org.nd4j.linalg.dataset DataSet)
           (org.deeplearning4j.nn.conf.layers DenseLayer DenseLayer$Builder OutputLayer OutputLayer$Builder)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.nd4j.linalg.activations Activation)
           (clojure.lang Reflector)))

(def seed 123)
(def rate 0.01)
(def numOutputs 2)
(def batchSize 50)
(def nEpochs 30)

(Reflector/invokeConstructor
  (resolve (symbol "Integer"))
  (to-array ["16"]))

;(-> "classification/linear_data_eval.csv" io/resource io/file FileSplit.)

(defn resource->records-iter [resource-path]
  (RecordReaderDataSetIterator.
    (doto (CSVRecordReader.)
      (.initialize (-> resource-path io/resource io/file FileSplit.)))
    batchSize 0 2))

(def trainIter (resource->records-iter "classification/linear_data_train.csv"))
(def testIter (resource->records-iter "classification/linear_data_eval.csv"))

(defn layer-resolver [layer-type]
  (-> layer-type
      name
      (cs/split #"-")
      (into ["Layer" "$" "Builder"])
      (->> (map (fn [s] (cond-> s (not (re-matches #"\d+D" s)) cs/capitalize))))
      cs/join
      symbol
      resolve))

(defn build-layer [layer-type args]
  (-> layer-type layer-resolver (Reflector/invokeConstructor (to-array args))))

(eval (cons 'new (list (layer-resolver :dense))))

(defmacro kw->enum [pre k]
  (-> k name cs/upper-case (cs/replace #"-" "_") (->> (str pre)) symbol))

(defmacro weight [kw] `(kw->enum "WeightInit/" ~kw))
(defmacro activation [kw] `(kw->enum "Activation/" ~kw))
(defmacro loss [kw] `(kw->enum "LossFunctions$LossFunction/" ~kw))

(defmacro layer [layer-type args & {:keys [in out weight activation]}]
  `(let [builder# (new ~(layer-resolver layer-type) ~@args)]
     (.build
       (cond-> builder#
               ~in (.nIn ~in)
               ~out (.nOut ~out)
               ~weight (.weightInit (weight ~weight))
               ~activation (.activation (activation ~activation))))))

(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed seed)
      (.iterations 1)
      (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (.learningRate rate)
      (.updater Updater/NESTEROVS)
      (.list)
      (.layer 0 (layer :dense [] :in 2 :out 20 :weight :xavier :activation :relu))
      (.layer 1 (layer :output [(loss :negativeloglikelihood)]
                       :in 20 :out 2 :weight :xavier :activation :softmax))
      (.pretrain false)
      (.backprop true)
      (.build)))

(def model (doto (MultiLayerNetwork. conf)
             (.init)
             (.setListeners [(ScoreIterationListener. 10)])))

(dotimes [n nEpochs]
  (.fit model trainIter))

(println "Evaluate model....")

(let [eval (Evaluation. numOutputs)]
  (while (.hasNext testIter)
    (let [^DataSet t (.next testIter)]
      (.eval eval (.getLabels t) (.output model (.getFeatureMatrix t) false))))
  (println (.stats eval)))

(defn skewer-case->JAVA_CASE [s]
  (-> s name cs/upper-case (cs/replace #"-" "_")))

(defmulti lookup (comp keyword namespace))

(defmethod lookup :updater [s]
  (-> s skewer-case->JAVA_CASE Updater/valueOf))

(defmethod lookup :weight-init [s]
  (-> s skewer-case->JAVA_CASE WeightInit/valueOf))

(defmethod lookup :optimization-algorithm [s]
  (-> s skewer-case->JAVA_CASE OptimizationAlgorithm/valueOf))

(defmethod lookup :activation [s]
  (-> s skewer-case->JAVA_CASE Activation/valueOf))

(defmethod lookup :loss [s]
  (-> s skewer-case->JAVA_CASE LossFunctions$LossFunction/valueOf))

(defmethod lookup :default [s] s)

(defmulti bar (fn [k _] k))

(defmethod bar :activation [_ s]
  (-> s skewer-case->JAVA_CASE Activation/valueOf))

(defmethod bar :weight-init [_ s]
  (-> s skewer-case->JAVA_CASE WeightInit/valueOf))

(Reflector/invokeInstanceMember
  "weightInit"
  (Reflector/invokeInstanceMember
    "activation"
    (layer-builder :dense nil)
    (object-array [(lookup :activation/relu)]))
  (object-array [(lookup :weight-init/xavier)]))

{:seed 1239
 :iterations 1
 :optimization-algorithm :optimization-algorithm/stochastic-gradient-descent
 :learning-rate 0.013
 :updater :updater/nesterovs
 :layers [[:layer/dense []
           :nodes 2
           :weight :weight-init/xavier
           :activation :activation/relu]
          [:layer/output [:loss/negativeloglikelihood]
           :nodes 20
           :weight :weight-init/xavier
           :activation :activation/softmax]]
 :pretrain  false
 :backprop  true}

;(defmacro params [conf & params]
;  `(cond-> (NeuralNetConfiguration$Builder.)
;          ~@(mapcat (fn [p] `((~p ~conf) (~(symbol (str "." (name p))) (~p ~conf)))) params)))
;
;(defn netconf [conf]
;  (params conf :seed :iterations :updater :pretrain :backprop))
;
;(netconf
;  {:seed 1239
;   :iterations 1
;   :optimization-algorithm :stochastic-gradient-descent
;   :learning-rate 0.013
;   :updater :nesterovs
;   :layers [(layer :dense [] :in 2 :out 20 :weight :xavier :activation :relu)
;            (layer :output [(loss :negativeloglikelihood)]
;                   :in 20 :out 2 :weight :xavier :activation :softmax)]
;   :pretrain false
;   :backprop true})