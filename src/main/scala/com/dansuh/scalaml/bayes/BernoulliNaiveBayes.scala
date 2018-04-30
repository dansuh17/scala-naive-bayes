package com.dansuh.scalaml.bayes

trait BernoulliNaiveBayes {
  def predictedClass(data: Seq[(Int, Seq[Int])], x: Seq[Int]): Int = {
    BernoulliNaiveBayes.predict(
      BernoulliNaiveBayes.likelihood(data),
      BernoulliNaiveBayes.prior(data),
      x
    )
  }
}

object BernoulliNaiveBayes {
  // LabelData type : (y, [x1, x2, ... xn])
  type LabelData = Seq[(Int, Seq[Int])]
  // we want to calculate P(Y | X) sets
  // Map[Y = y, (P(X1 = true | Y1 = y), P(X2 = true | Y2 = y), ...)]
  // here the assumption is that X is binary
  def likelihood(data: LabelData): Map[Int, Seq[Double]] = {
    val classes: Seq[Int] = data.map(_._1).distinct.sorted
    val groupedByClass: Map[Int, Seq[(Int, Seq[Int])]] =
      data.groupBy(_._1)

    // returns the counts per attributes that are eqiual to xVal
    def counts(data: Seq[(Int, Seq[Int])], xVal: Int): Seq[Int] = {
      val numX = data.head._2.length  // number of attributes
      (0 to numX) map (attrNum => data.count(_._2(attrNum) == xVal))
    }

    val allCountsX: Seq[Seq[Int]] =
      classes map (groupedByClass(_)) map (counts(_, 1))

    val classCounts: Seq[Int] = classes.map(groupedByClass(_)).map(_.size)

    val condProbs: Seq[Seq[Double]] =
      allCountsX.zip(classCounts).seq.map(ec => ec._1.map(_.toDouble / ec._2))

    classes.zip(condProbs).toMap
  }

  // calculate the priors P(Y = y)
  def prior(data: LabelData): Vector[(Int, Float)] =
    data.foldLeft(Vector[(Int, Int)]())((acc, datPoint) => {
      // TODO: VERY unnecessary repetitive computation - can it be avoided?
      val exists = acc.forall(_._1 != datPoint._1)
      if (exists) {
        // modify only that part
        val (front, back) = acc span (_._1 != datPoint._1)
        val newCount = (back.head._1, back.head._2 + 1)
        front ++ (newCount +: back)
      } else {  // create a new class and set count to 1
        acc :+ (datPoint._1, 1)
      }
    }).map(c => (c._1, c._2.toFloat / data.size))  // normalize to probabilities


  // calculate argmax P(Y = y | X) = argmax P(Y) * P(X | Y)
  // where P(X | Y) = prod(P(x_i | Y))
  def predict(likelihood: Map[Int, Seq[Double]],
              prior: Vector[(Int, Float)],
              new_x: Seq[Int]): Int = {
    val classes: Seq[Int] = (prior map {_._1}).distinct.sorted
    val probs: Seq[Double] = classes.map(likelihood(_).zip(new_x).map(item => {
      val (prob, xVal) = item
      if (xVal == 1) prob
      else 1 - prob
    })).map(_.foldRight(1d)(_ * _))
    val argmax: Int = probs.zipWithIndex.maxBy(_._1)._2
    argmax
  }
}
