package com.dansuh.scalaml.bayes

trait BernoulliNaiveBayes {
}

object BernoulliNaiveBayes {
  // we want to calculate P(Y | X) sets
  // Map[Y = y, (P(X1 = true | Y1 = y), P(X2 = true | Y2 = y), ...)]
  // here the assumption is that X is binary
  def likelihood(data: Seq[(Int, Seq[Int])]): Map[Int, Seq[Double]] = {
    val classes: Seq[Int] = data.map(_._1).distinct.sorted
    // TODO:
  }

  // calculate the priors P(Y = y)
  def prior(data: Seq[(Int, Seq[Int])]): Vector[(Int, Float)] =
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


  def predict(probability: Map[Int, Seq[Double]], new_x: Seq[Int]): Int = {
    // TODO
  }
}
