package com.dansuh.scalaml.data

import java.io.BufferedInputStream
import java.util.zip.GZIPInputStream

// case classes for MNIST
case class MnistData(_1: Int, _2: Seq[Float]) extends LabeledData[Int, Seq[Float]]
case class MnistDataSet(data: Seq[LabeledData[Int, Seq[Float]]])
  extends DataSet[Int, Seq[Float]] {
  def mean(list: Seq[Float]): Float = list.sum / list.size
  def sampleMeans: Map[Int, Seq[Float]] =
    groupByClass.mapValues(valList => {
      val samples: Seq[Seq[Float]] = valList.map(_.sample)
      samples.transpose.map(lst => lst.sum / lst.size)
    })
  def sampleVariances: Map[Int, Seq[Float]] =
}

object Mnist {
  @inline def byteToUnsignedInt(b: Byte): Int = b.toInt & 0xFF

  private lazy val trainImagesFile = getFileStream("train-images-idx3-ubyte")
  private lazy val trainLabelsFile = getFileStream("train-labels-idx1-ubyte")
  lazy val trainImagesByteArray: Array[Array[Byte]] = readImages(trainImagesFile)
  lazy val trainLabels: Stream[Int] = readLabels(trainLabelsFile)

  def trainImages: Iterator[Vector[Float]] =
    trainImagesByteArray.iterator.map(_.toVector.map(byteToUnsignedInt(_).toFloat))

  def labeledTrainIterator: Iterator[(Int, Vector[Float])] =
    trainLabels.iterator zip trainImages

  private def getFileStream(name: String): Iterator[Int] = {
    val inStream = new BufferedInputStream(new GZIPInputStream(
      // TODO: more understanding
      this.getClass.getClassLoader.getResourceAsStream(s"mnist/$name.gz")
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

  // final output
  val minstDataSet: MnistDataSet =
    MnistDataSet(DataSet fromTupleSequence labeledTrainIterator.toSeq)
}
