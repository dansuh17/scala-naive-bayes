package com.dansuh.scalaml.data

trait LabeledData[LabelType, SampleType] extends Product2[LabelType, SampleType] {
  def label: LabelType = _1
  def sample: SampleType = _2
}

object LabeledData {
  def apply[A, B](l: A, s: B): LabeledData[A, B] = new LabeledData[A, B] {
    override def _1: A = l
    override def _2: B = s
    override def canEqual(that: Any): Boolean = that.isInstanceOf[LabeledData[A, B]]
  }
}

trait DataSet[Label, Sample] {
  // type alias for sequence of labeled data
  type DataSeqType = Seq[LabeledData[Label, Sample]]

  // the actual data
  def data: Seq[LabeledData[Label, Sample]]
  def size: Int = data.size

  // represents a sequence of labeled data
  def classes: Seq[Label] = data.map(_.label).distinct

  // groups samples by their classes
  lazy val groupByClass: Map[Label, Seq[Sample]] =
    data groupBy(_.label) mapValues(_.map(_.sample))

  // given a sample, return the label
  def getLabel(sample: Sample): Option[Label] = {
    val dropped = data.dropWhile(_.sample == sample)
    if (dropped.isEmpty) None
    else Some(dropped.head.label)
  }
}

object DataSet {
  // construct DataSet from sequence of tuples
  def fromTupleSequence[A, B](rawData: Seq[(A, B)]): Seq[LabeledData[A, B]] =
    rawData.map(a => LabeledData(a._1, a._2))
}
