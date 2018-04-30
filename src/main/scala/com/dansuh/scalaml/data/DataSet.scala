package com.dansuh.scalaml.data

trait LabeledData[LabelType, SampleType] extends Product2[LabelType, SampleType] {
  def label: LabelType = _1
  def sample: SampleType = _2
}

trait DataSet[Label, Sample] {
  // type alias for sequence of labeled data
  type DataSeqType = Seq[LabeledData[Label, Sample]]

  // represents the actual data
  def data: Seq[LabeledData[Label, Sample]]
  // represents a sequence of labeled data
  def classes: Seq[Label] = data.map(_.label).distinct

  def groupByClass: Map[Label, DataSeqType] = data.groupBy(_.label)

  def getLabel(sample: Sample): Option[Label] = {
    val dropped = data.dropWhile(_.sample == sample)

    if (dropped.isEmpty) None
    else Some(dropped.head.label)
  }
}

case class MnistData(data: LabeledData[Int, Seq[Int]])

object DataSet {
}
