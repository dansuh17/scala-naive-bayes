package com.dansuh.scalaml.data

object Main {
  def main(args: Array[String]): Unit = {
    val mnist: MnistDataSet = Mnist.minstDataSet
    println(mnist.size)
    println(mnist.sampleMeans)
    println(mnist.sampleVariances)
  }
}
