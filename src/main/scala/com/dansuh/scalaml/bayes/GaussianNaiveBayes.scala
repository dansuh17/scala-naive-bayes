package com.dansuh.scalaml.bayes

import com.dansuh.scalaml.data.{DataSet, MnistDataSet}

trait GaussianNaiveBayes {
  def data: DataSet[Int, Seq[Float]]

  // calculate P(X | Y)
  def likelihood: Map[Int, Seq[Float => Double]]

  // calculate P(Y)
  def prior: Map[Int, Float]

  // calculate P(Y | X)
  def predict(x: Seq[Float]): Int
}

case class MnistGaussianNaiveBayesClassifier(data: MnistDataSet)
  extends GaussianNaiveBayes {
  def gaussianProb(mean: Float, variance: Float)(x: Float): Double = {
    (1 / math.sqrt(2 * math.Pi * variance)) * math.pow(math.E, -(math.pow(x - mean, 2) / (2 * variance)))
  }
  override def likelihood: Map[Int, Seq[Float => Double]] = data.sampleMeanVariances.mapValues(_.map({ case (mean, variance) => {
    gaussianProb(mean, variance)
  }}))

  override def prior: Map[Int, Float] = {
    lazy val totalSize = data.size
    data.groupByClass.mapValues(_.size.toFloat / totalSize)
  }

  // def argmax[A](list: Seq[A]): Int = list.view.zipWithIndex.maxBy(_._1)._2
  def argmax[A](map: Map[A, Double]): A = map.keysIterator.reduceLeft((x, y) => if (map(x) > map(y)) x else y)

  override def predict(x: Seq[Float]): Int = argmax(prior.map({
    case (cls, pri) =>
      val like = likelihood(cls).zip(x).map({ case (f, xVal) => f(xVal) }).product * pri
      cls -> like
  }))
}
