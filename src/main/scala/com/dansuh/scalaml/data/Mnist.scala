package com.dansuh.scalaml.data

import java.io.BufferedInputStream
import java.util.zip.GZIPInputStream

// case classes for MNIST
case class MnistData(_1: Int, _2: Seq[Float]) extends LabeledData[Int, Seq[Float]]
case class MnistDataSet(data: Seq[LabeledData[Int, Seq[Float]]])
  extends DataSet[Int, Seq[Float]] {
  /**
    * Find the mean of values.
    * @param list sequence of values
    * @return mean value
    */
  def mean(list: Seq[Float]): Float = list.sum / list.size

  def variance(list: Seq[Float], mean: Float): Float =
    (list.map(x => math.pow(x - mean, 2)).sum / list.size).toFloat

  def variance(list: Seq[Float]): Float = {
    val avg = mean(list)
    variance(list, avg)
  }

  /**
    * Return a map of sample means.
    * (Y = y) => (mean(X1), mean(X2), ...)
    * @return map corresponding to means of each class
    */
  lazy val sampleMeans: Map[Int, Seq[Float]] =
    groupByClass.mapValues(_.transpose.map(mean))

  /**
    * Return sample variances.
    * (Y = y) => (variance(X1), variance(X2), ...)
    * @return map corresponding to variances of each class to each attributes
    */
  lazy val sampleVariances: Map[Int, Seq[Float]] =
    groupByClass.mapValues(_.transpose.map(variance))

  // given a list, calculate both the mean and variance
  def meanVariance(list: Seq[Float]): (Float, Float) = {
    lazy val avg = mean(list)
    (avg, variance(list, mean = avg))
  }

  lazy val sampleMeanVariances: Map[Int, Seq[(Float, Float)]] =
    // here the values are sequence of Seq[Float], whose length is 784 (for MNIST)
    // transpose of this will make a length 784-sequence, grouping the samples by attributes
    groupByClass.mapValues(_.transpose.map(meanVariance).toVector)
}

object Mnist {
  @inline def byteToUnsignedInt(b: Byte): Int = b.toInt & 0xFF

  private lazy val trainImagesFile = getFileStream("train-images-idx3-ubyte")
  private lazy val trainLabelsFile = getFileStream("train-labels-idx1-ubyte")
  private lazy val testImagesFile = getFileStream("t10k-images-idx3-ubyte")
  private lazy val testLabelsFile = getFileStream("t10k-labels-idx1-ubyte")

  lazy val trainImagesByteArray: Array[Array[Byte]] = readImages(trainImagesFile)
  lazy val trainLabels: Stream[Int] = readLabels(trainLabelsFile)
  lazy val testImagesByteArray: Array[Array[Byte]] = readImages(testImagesFile)
  lazy val testLabels: Stream[Int] = readLabels(testLabelsFile)

  def trainImages: Iterator[Vector[Float]] =
    trainImagesByteArray.iterator.map(_.toVector.map(byteToUnsignedInt(_).toFloat))
  def testImages: Iterator[Vector[Float]] =
    testImagesByteArray.iterator.map(_.toVector.map(byteToUnsignedInt(_).toFloat))

  def labeledTrainIterator: Iterator[(Int, Vector[Float])] =
    trainLabels.iterator zip trainImages
  def labeledTestIterator: Iterator[(Int, Vector[Float])] =
    testLabels.iterator zip testImages

  private def getFileStream(name: String): Iterator[Int] = {
    val inStream = new BufferedInputStream(new GZIPInputStream(
      // TODO: more understanding
      this.getClass.getClassLoader.getResourceAsStream(s"$name.gz")
    ))
    // -1 if EOF
    Iterator.continually(inStream.read()).takeWhile(_ != -1)
  }

  private val ImagesOffset = 16
  private val LabelsOffset = 8
  val ImageWidth = 28
  val ImageHeight = 28
  private val ImageSize = ImageWidth * ImageHeight

  private def readImages(bytes: Iterator[Int]): Array[Array[Byte]] = {
    bytes.drop(ImagesOffset)  // magic number + set size
      .map(_.toByte).grouped(ImageSize).map(_.toArray).toArray
  }
  // label stream
  private def readLabels(bytes: Iterator[Int]): Stream[Int] =
    bytes.drop(LabelsOffset).toStream

  // this is what we will use
  val minstDataSet: MnistDataSet =
    MnistDataSet(DataSet fromTupleSequence labeledTrainIterator.toSeq)
  val mnistTestDataSet: MnistDataSet =
    MnistDataSet(DataSet fromTupleSequence labeledTestIterator.toSeq)
}
