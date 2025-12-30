package dimwit.autodiff

import dimwit.*
import dimwit.tensor.*
import dimwit.Conversions.given
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import org.scalacheck.Gen
import me.shadaj.scalapy.py

import TensorGen.*
import TestUtil.*

class AutodiffSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  property("derivative scalar function x^2"):
    def f(x: Tensor0[Float]) = x * x
    def dfTruth(x: Tensor0[Float]): Tensor0[Float] = x * 2.0f
    val df = Autodiff.grad(f)
    forAll(tensor0Gen(VType[Float])): x =>
      df(x) shouldBe dfTruth(x)

  property("second-order derivative scalar function x^2"):
    def f(x: Tensor0[Float]) = x * x
    def ddfTruth(x: Tensor0[Float]): Tensor0[Float] = 2.0f
    val ddf = Autodiff.grad(Autodiff.grad(f))

    forAll(tensor0Gen(VType[Float])): x =>
      ddf(x) should approxEqual(ddfTruth(x))

  property("third-order derivative scalar function x^2"):
    def f(x: Tensor0[Float]) = x * x
    def dddfTruth(x: Tensor0[Float]): Tensor0[Float] = 0.0f
    val dddf = Autodiff.grad(Autodiff.grad(Autodiff.grad(f)))

    forAll(tensor0Gen(VType[Float])): x =>
      dddf(x) should approxEqual(dddfTruth(x))

  property("derivative vector function sum(x^2)"):
    def f(x: Tensor1[A, Float]) = (x * x).sum
    def dfTruth(x: Tensor1[A, Float]): Tensor1[A, Float] = x *! 2.0f
    val df = Autodiff.grad(f)
    forAll(tensor1Gen(VType[Float])): x =>
      df(x) should approxEqual(dfTruth(x))

  property("derivative two parameter function sum((x+y*2)^2)"):
    def f(x: Tensor1[A, Float], y: Tensor1[A, Float]) = ((x + (y *! 2f)).pow(2)).sum
    def dfTruth(x: Tensor1[A, Float], y: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float]) =
      // TODO :* has lower precedence than +, so need parentheses here, maybe syntax must be changed?
      val xGrad = (x *! 2.0f) + (y *! 4.0f)
      val yGrad = (x *! 4.0f) + (y *! 8.0f)
      (xGrad, yGrad)
    val df = Autodiff.grad(f)
    forAll(twoTensor1Gen(VType[Float])): (x, y) =>
      val (xGrad, yGrad) = df(x, y)
      val (xGradTruth, yGradTruth) = dfTruth(x, y)
      xGrad should approxEqual(xGradTruth)
      yGrad should approxEqual(yGradTruth)

  property("jacobian vector function f(x) = 2x"):
    val n = 2
    def f(x: Tensor1[A, Float]) = x *! 2.0f
    def dfTruth(x: Tensor1[A, Float]): Tensor2[A, A, Float] =
      Tensor2.eye(x.dim(Axis[A]), VType[Float]) *! 2.0f
    val jf = Autodiff.jacobian(f)
    forAll(tensor1Gen(VType[Float])): x =>
      jf(x) should approxEqual(dfTruth(x))
