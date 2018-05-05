package com.dansuh.scalaml.data

import com.dansuh.scalaml.bayes.MnistGaussianNaiveBayesClassifier

object Main {
  def main(args: Array[String]): Unit = {
    val mnist: MnistDataSet = Mnist.minstDataSet
    val mnistTest: MnistDataSet = Mnist.mnistTestDataSet
    val testData = mnistTest.data
    println(mnist.size)
    // println(mnist.sampleMeans)
    // println(mnist.sampleVariances)
    val classifier = MnistGaussianNaiveBayesClassifier(mnist)
    println("classifier loaded -calculating priors")
    val priors: Map[Int, Float] = classifier.prior
    println("calculating likelihoods")
    val likelihoods: Map[Int, Seq[Float => Double]] = classifier.likelihood
    println(likelihoods)
    // val predictedClass = classifier.predict(testData.head.sample)
    // println(predictedClass)
    // println(testData.head.label)
  }
}
