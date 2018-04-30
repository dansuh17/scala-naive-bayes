package com.dansuh.scalaml.data

trait LabeledData[LabelType, SampleType] extends Product2[LabelType, SampleType] {
  def label: LabelType = _1
  def sample: SampleType = _2
}

object LabeledData {
  def apply[A, B](l: A, s: B) = new LabeledData[A, B](l, s) {
    override def _1: A = l
    override def _2: B = s
    override def canEqual(that: Any): Boolean = that.isInstanceOf[LabeledData[A, B]]
  }
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

case class MnistData(_1: Int, _2: Seq[Int]) extends LabeledData[Int, Seq[Int]]
case class MnistDataSet(data: Seq[LabeledData[Int, Seq[Int]]])
  extends DataSet[Int, Seq[Int]]

object DataSet {
  def fromTupleSequence[A, B](rawData: Seq[(A, B)]): DataSet[A, B] = new DataSet[A, B] {
    def data: Seq[LabeledData[A, B]] = rawData.map(a => LabeledData(a._1, a._2))
  }
}