package src.main.scala.basic

import dimwit.*
import dimwit.autodiff.*

object Playground extends App:
  val k = Key(42)

  trait A derives Label
  trait B derives Label

  def f(x: Tensor1[A, Float]): Tensor0[Float] =
    x.sum

  grad(f)
