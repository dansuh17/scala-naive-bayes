package com.dansuh.scalaml.bayes

import com.dansuh.scalaml.data.{DataSet, MnistDataSet}

trait GaussianNaiveBayes {
  def data: DataSet[Int, Seq[Float]]

  // calculate P(X | Y)
  def logLikelihood: Map[Int, Seq[Float => Option[Double]]]

  // calculate P(Y)
  def logPrior: Map[Int, Float]

  // calculate P(Y | X)
  def predict(x: Seq[Float]): Int
}

case class MnistGaussianNaiveBayesClassifier(data: MnistDataSet)
  extends GaussianNaiveBayes {

  // returns the probability of x in Gaussian distribution
  def gaussianProb(mean: Float, variance: Float)(x: Float): Option[Double] = {
    val multiplier: Double = 1.0f / math.sqrt(2.0f * math.Pi * variance)
    val denom = 2.0f * variance
    val exponent: Double = -(math.pow(x - mean, 2) / denom)
    val logprob = math.log(multiplier * math.pow(math.E, exponent))

    if (logprob.isNaN) None
    else Some(logprob)
  }

  val epsilon = 0.1f
  lazy val logLikelihood: Map[Int, Seq[Float => Option[Double]]] =
    // epsilon is added as a prior (hence making this an MAP estimate)
    data.sampleMeanVariances.mapValues(_.map({ case (mean, variance) =>
      gaussianProb(mean, variance + epsilon) _
    }))

  lazy val logPrior: Map[Int, Float] = {
    lazy val totalSize = data.size
    data.groupByClass.mapValues(x => math.log(x.size.toFloat / totalSize).toFloat)
  }

  // def argmax[A](list: Seq[A]): Int = list.view.zipWithIndex.maxBy(_._1)._2
  def argmax[A](map: Map[A, Double]): A =
    map.keysIterator.reduceLeft((x, y) => if (map(x) > map(y)) x else y)

  override def predict(x: Seq[Float]): Int = argmax(logPrior.map({
    case (cls, pri) =>
      val like = logLikelihood(cls).zip(x).flatMap { case (f, xVal) => f(xVal) }
      val posterior = like.sum + pri
      cls -> posterior
  }))
}
