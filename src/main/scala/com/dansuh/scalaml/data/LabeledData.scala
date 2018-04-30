package com.dansuh.scalaml.data

trait LabeledData[LabelType, DataType] {
  // represents a sequence of labeled data
  type Data = Seq[(LabelType, Seq[DataType])]

  def classes(data: Data): Seq[LabelType] = data.map(_._1).distinct

  def groupByClass(data: Data): Map[LabelType, Data] = data.groupBy(_._1)
}

object LabeledData {

}
