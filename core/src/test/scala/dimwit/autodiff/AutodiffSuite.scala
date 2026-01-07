package dimwit.autodiff

import dimwit.*
import dimwit.tensor.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

import TestUtil.*
import dimwit.autodiff.Autodiff.Gradient

class AutodiffSuite extends AnyFunSpec with Matchers:

  describe("grad"):
    describe("single parameter function"):
      it("d¹, d², d³ of x²"):
        def f(x: Tensor0[Float]) = x * x
        val df = Autodiff.grad(f)
        val ddf = Autodiff.grad(df)
        val dddf = Autodiff.grad(ddf)

        val x = Tensor0(3.0f)
        df(x) shouldEqual Tensor0(6.0f)
        ddf(x) shouldEqual Tensor0(2.0f)
        dddf(x) shouldEqual Tensor0(0.0f)

      it("d¹ sum(x²)"):
        def f(x: Tensor1[A, Float]) = (x * x).sum
        val df = Autodiff.grad(f)

        val x = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 5.0f))
        df(x) shouldEqual Tensor1.fromArray(Axis[A], VType[Float])(Array(2.0f, 10.0f))

      it("d¹ function using vmap"):
        def f(x: Tensor2[A, B, Float]) = x.vmap(Axis[A])(_.sum).sum
        val df = Autodiff.grad(f)

        val x = Tensor.ones(Shape(Axis[A] -> 2, Axis[B] -> 2), VType[Float])
        df(x) shouldEqual Tensor.ones(x.shape, x.vtype)

    describe("two parameter function"):
      it("d¹/dx and d¹/dy of (x + 2y)²"):
        def f(x: Tensor1[A, Float], y: Tensor1[A, Float]) = ((x + (y *! 2.0f)).pow(Tensor0(2.0f))).sum
        val df = Autodiff.grad(f)

        val x = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f))
        val y = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f))

        val (xGrad, yGrad) = df(x, y)
        xGrad shouldEqual Tensor1.fromArray(Axis[A], VType[Float])(Array(6.0f))
        yGrad shouldEqual Tensor1.fromArray(Axis[A], VType[Float])(Array(12.0f))

  describe("jacobian"):
    describe("single parameter function"):
      it("Jacobian of f: R² -> R², f(x) = 2x"):
        def f(x: Tensor1[A, Float]) = x *! 2.0f
        val jf = Autodiff.jacobian(f)

        val x = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 1.0f))
        jf(x) should approxEqual(Tensor2.eye(x.dim(Axis[A]), x.vtype) *! 2.0f)

  describe("jacRev / jacFwd"):

    // setup engines to test both modes in the same way
    val engines = List(
      ("jacRev", [In: ToPyTree, Out: ToPyTree] => (f: In => Out) => (gradTree: ToPyTree[Gradient[In, Out]]) ?=> Autodiff.jacRev[In, Out](f)),
      ("jacFwd", [In: ToPyTree, Out: ToPyTree] => (f: In => Out) => (gradTree: ToPyTree[Gradient[In, Out]]) ?=> Autodiff.jacFwd[In, Out](f))
    )

    engines.foreach:
      case (modeName, jacMode) =>
        it(s"$modeName d¹ on f: R² -> R², f(x) = swap(x)"):
          def f(x1: Tensor1[A, Float], x2: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float]) = (x2, x1)
          val df = jacMode(f.tupled)
          val x1 = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 0.0f))
          val x2 = Tensor1.fromArray(Axis[A], VType[Float])(Array(0.0f, 1.0f))
          val (x1Grad, x2Grad) = df(x1, x2)
          val (x1_dx1, x1_dx2) = x1Grad
          val (x2_dx1, x2_dx2) = x2Grad
          x1_dx1 should approxEqual(Tensor.zeros(x1_dx1.shape, x1_dx1.vtype))
          x1_dx2 should approxEqual(Tensor2.eye(x1.dim(Axis[A]), x1.vtype))
          x2_dx1 should approxEqual(Tensor2.eye(x2.dim(Axis[A]), x2.vtype))
          x2_dx2 should approxEqual(Tensor.zeros(x2_dx2.shape, x2_dx2.vtype))

        it(s"$modeName d² on f: R² -> R, f(x1, x2) = sum(x1 * x2)"):
          def f(x1: Tensor1[A, Float], x2: Tensor1[A, Float]): Tensor0[Float] = (x1 * x2).sum
          val df = jacMode(f.tupled)
          val ddf = jacMode(df)
          val x1 = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 2.0f))
          val x2 = Tensor1.fromArray(Axis[A], VType[Float])(Array(3.0f, 4.0f))
          val (x1Grad, x2Grad) = ddf(x1, x2)
          val (x1_dx1, x1_dx2) = x1Grad
          val (x2_dx1, x2_dx2) = x2Grad
          x1_dx1 should approxEqual(Tensor.zeros(x1_dx1.shape, x1_dx1.vtype))
          x1_dx2 should approxEqual(Tensor2.eye(x1.dim(Axis[A]), x1.vtype) *! Tensor0(1.0f))
          x2_dx1 should approxEqual(Tensor2.eye(x2.dim(Axis[A]), x2.vtype) *! Tensor0(1.0f))
          x2_dx2 should approxEqual(Tensor.zeros(x2_dx2.shape, x2_dx2.vtype))

  describe("Complex application"):
    it("case class support"):
      case class Params(w: Tensor1[A, Float], b: Tensor0[Float])
      def loss(data: Tensor1[A, Float])(params: Params): Tensor0[Float] =
        ((data * params.w).sum + params.b).pow(Tensor0(2.0f))
      val trainData = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 2.0f))
      val dloss = Autodiff.grad(loss(trainData))
      val params = Params(Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 2.0f)), Tensor0(3.0f))
      val dParams = dloss(params)
      dParams.w shouldEqual Tensor1.fromArray(Axis[A], VType[Float])(Array(16.0f, 32.0f))
