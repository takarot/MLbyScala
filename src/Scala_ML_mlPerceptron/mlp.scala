package Scala_ML_mlPerceptron

/**
  * Created by takanori on 2016/07/25.
  */

object mlp {
  def main(args: Array[String]) = {

    //排他的論理和(xor)
    val samples = Seq (
      (Seq(0.0, 0.0), Seq(0.0)),
      (Seq(0.0, 1.0), Seq(1.0)),
      (Seq(1.0, 0.0), Seq(1.0)),
      (Seq(1.0, 1.0), Seq(0.0))
    )

    val m = new mlp(samples, 3,3,1)

    println(m.pred(Seq(0.0,0.0)))
    println(m.pred(Seq(0.0,1.0)))
    println(m.pred(Seq(1.0,0.0)))
    println(m.pred(Seq(1.0,1.0)))
  }
}

class mlp (samples: Seq[(Seq[Double], Seq[Double])], I: Int, J: Int, K: Int){

  val w = Seq(Array.fill(I, J)(Math.random()), Array.fill(J, K)(Math.random()))

  for(step <- 1 to 10000; (x1, tn) <- samples) {
    val (x2, x3) = (prop(x1, 0), pred(x1))
    back(back(tn zip x3 map{case (a, x) => x - a}, x2, x3, 1), x1, x2, 0)
  }

  def prop(x: Seq[Double], n: Int) = for(j <- w(n).head.indices) yield {
    1 / (1 + Math.exp(-( w(n) zip x).map{case (wnij, xi)=> wnij(j) * xi}.sum))
  }

  def pred(input: Seq[Double]) = prop(prop(input, 0), 1)

  def back(e: Seq[Double], x: Seq[Double], y: Seq[Double], n: Int) = {
    val g = e zip y map{case (e, y) => y * (1 - y) * e}
    for(i <- x.indices; j <- y.indices) w(n)(i)(j) -= 0.01 * g(j) * x(i)
    for(i <- x.indices) yield (g zip w(n)(i)).map{case (g, w) => g * w}.sum
  }
}
