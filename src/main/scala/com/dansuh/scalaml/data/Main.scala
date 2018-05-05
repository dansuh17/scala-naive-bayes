package com.dansuh.scalaml.data

import com.dansuh.scalaml.bayes.MnistGaussianNaiveBayesClassifier

object Main {
  def main(args: Array[String]): Unit = {
    val mnist: MnistDataSet = Mnist.minstDataSet
    val mnistTest: MnistDataSet = Mnist.mnistTestDataSet
    val testData = mnistTest.data
    println(mnist.size)
    val classifier = MnistGaussianNaiveBayesClassifier(mnist)
    println("classifier loaded -calculating prior estimates")
    val priors: Map[Int, Float] = classifier.logPrior
    println("calculating likelihood estimates")
    val likelihoods: Map[Int, Seq[Float => Option[Double]]] = classifier.logLikelihood
    println(likelihoods)
    println("calculating means and variances")
    println(mnist.sampleMeanVariances)
    val predictedClass = classifier.predict(testData.head.sample)
    println(predictedClass)
    println(testData.head.label)
    println("Testing with 10 examples:")
    val predictTestSet = testData.take(10).map(samp => (samp.label, classifier.predict(samp.sample)))
    println("Results: (correct_label, predicted_label)")
    println(predictTestSet.toVector)
  }
}
