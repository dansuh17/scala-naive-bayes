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

  // returns the probability of x in Gaussian distribution
  def gaussianProb(mean: Float, variance: Float)(x: Float): Double = {
    val exponent: Double = -(math.pow(x - mean, 2) / (2.0f * variance))
    (1.0f / math.sqrt(2.0f * math.Pi * variance)) * math.pow(math.E, exponent)
  }

  lazy val likelihood: Map[Int, Seq[Float => Double]] =
    data.sampleMeanVariances.mapValues(_.map({ case (mean, variance) =>
      gaussianProb(mean, variance) _
    }))

  lazy val prior: Map[Int, Float] = {
    lazy val totalSize = data.size
    data.groupByClass.mapValues(_.size.toFloat / totalSize)
  }

  // def argmax[A](list: Seq[A]): Int = list.view.zipWithIndex.maxBy(_._1)._2
  def argmax[A](map: Map[A, Double]): A =
    map.keysIterator.reduceLeft((x, y) => if (map(x) > map(y)) x else y)

  override def predict(x: Seq[Float]): Int = argmax(prior.map({
    case (cls, pri) =>
      val like = likelihood(cls).zip(x).map({ case (f, xVal) => f(xVal) }).product
      val posterior = like * pri

      cls -> posterior
  }))
}
