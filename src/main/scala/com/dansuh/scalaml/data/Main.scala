package com.dansuh.scalaml.data

import com.dansuh.scalaml.bayes.MnistGaussianNaiveBayesClassifier

object Main {
  def main(args: Array[String]): Unit = {
    val mnist: MnistDataSet = Mnist.minstDataSet
    val mnistTest: MnistDataSet = Mnist.mnistTestDataSet
    val testData = mnistTest.data
    val classifier = MnistGaussianNaiveBayesClassifier(mnist)
    println("classifier loaded -calculating prior estimates")
    println("Testing with 10 examples:")
    val predictTestSet = testData.take(10).map(samp => (samp.label, classifier.predict(samp.sample)))
    println("Results: (correct_label, predicted_label)")
    println(predictTestSet.toVector)
  }
}
