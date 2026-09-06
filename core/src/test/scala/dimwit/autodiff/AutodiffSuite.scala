package dimwit.autodiff

import dimwit.*
import dimwit.Conversions.given

/** A parameter tree, declared top level so its Mirror is available. */
case class JacParams(w: Tensor1[A, Float32], b: Tensor1[B, Float32]) derives TensorTree

class AutodiffSuite extends DimwitTest:

  describe("grad"):
    describe("single parameter function"):
      it("d¹, d², d³ of x²"):
        def f(x: Tensor0[Float32]) = x * x
        val df = Autodiff.grad(f)
        val ddf = Autodiff.grad((x: Tensor0[Float32]) => df(x).value)
        val dddf = Autodiff.grad((x: Tensor0[Float32]) => ddf(x).value)

        val x = Tensor0(3.0f)
        df(x) shouldEqual Tensor0(6.0f)
        ddf(x) shouldEqual Tensor0(2.0f)
        dddf(x) shouldEqual Tensor0(0.0f)

      it("d¹ sum(x²)"):
        def f(x: Tensor1[A, Float32]) = (x * x).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 5.0f))
        df(x) shouldEqual Tensor1(Axis[A]).fromArray(Array(2.0f, 10.0f))

      it("d¹ function using vmap"):
        def f(x: Tensor2[A, B, Float32]) = x.vmap(Axis[A])(_.sum).sum
        val df = Autodiff.grad(f)

        val x = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fill(1f)
        df(x) shouldEqual Tensor.like(x).fill(1f)

    describe("two parameter function"):
      it("d¹/dx and d¹/dy of (x + 2y)²"):
        def f(x: Tensor1[A, Float32], y: Tensor1[A, Float32]) = ((x + (y *! 2.0f)).pow(Tensor0(2.0f))).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f))
        val y = Tensor1(Axis[A]).fromArray(Array(1.0f))

        val (xGrad, yGrad) = df(x, y).value
        xGrad shouldEqual Tensor1(Axis[A]).fromArray(Array(6.0f))
        yGrad shouldEqual Tensor1(Axis[A]).fromArray(Array(12.0f))

  describe("valueAndGrad"):

    describe("two parameter function"):
      it("d¹/dx and d¹/dy of (x + 2y)²"):
        def f(x: Tensor1[A, Float32], y: Tensor1[A, Float32]) = ((x + (y *! 2.0f)).pow(Tensor0(2.0f))).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f))
        val y = Tensor1(Axis[A]).fromArray(Array(1.0f))

        val g = Autodiff.valueAndGrad(f)
        val (value, grad) = g(x, y)

        value shouldEqual f(x, y)
        grad shouldEqual df(x, y).value

  describe("jacobian"):
    describe("single parameter function"):
      it("Jacobian of f: R² -> R², f(x) = 2x"):
        def f(x: Tensor1[A, Float32]) = x *! 2.0f
        val jf = Autodiff.jacobian(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
        jf(x) should approxEqual(Tensor2(x.extent(Axis[A])).eye(x.vtype) *! 2.0f)

  describe("jacRev"):

    it("d¹ of f(x1, x2) = (x2, x1)"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): (Tensor1[A, Float32], Tensor1[A, Float32]) = (x2, x1)
      val df = Autodiff.jacRev(f.tupled)

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 0.0f))
      val x2 = Tensor1(Axis[A]).fromArray(Array(0.0f, 1.0f))
      val ((x1_dx1, x1_dx2), (x2_dx1, x2_dx2)) = df(x1, x2)

      // the first output is x2, so it depends on x2 only, and the other way round
      x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
      x1_dx2 should approxEqual(Tensor2(x1.extent(Axis[A])).eye(x1.vtype))
      x2_dx1 should approxEqual(Tensor2(x2.extent(Axis[A])).eye(x2.vtype))
      x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

    it("d¹ of f: Tensor1[A] => Tensor1[B] keeps the output axis first"):
      def f(x: Tensor1[A, Float32]): Tensor1[B, Float32] = x.relabel(Axis[A] -> Axis[B]) *! 2.0f
      val df = Autodiff.jacRev(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      df(x).axes shouldBe List("B", "A")
      df(x) should approxEqual((Tensor2(x.extent(Axis[A])).eye *! 2.0f).relabelAll((Axis[B], Axis[A])))

    it("d² of f(x1, x2) = sum(x1 * x2)"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] = (x1 * x2).sum
      val ddf = Autodiff.jacRev(Autodiff.jacRev(f.tupled))

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
      val ((x1_dx1, x1_dx2), (x2_dx1, x2_dx2)) = ddf(x1, x2)

      // d²/dx1² and d²/dx2² vanish, the mixed partials are the identity
      x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
      x1_dx2 should approxEqual(Tensor2(x1.extent(Axis[A])).eye(x1.vtype))
      x2_dx1 should approxEqual(Tensor2(x2.extent(Axis[A])).eye(x2.vtype))
      x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

  describe("jacFwd"):

    it("d¹ of f(x1, x2) = (x2, x1)"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): (Tensor1[A, Float32], Tensor1[A, Float32]) = (x2, x1)
      val df = Autodiff.jacFwd(f.tupled)

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 0.0f))
      val x2 = Tensor1(Axis[A]).fromArray(Array(0.0f, 1.0f))
      val ((x1_dx1, x1_dx2), (x2_dx1, x2_dx2)) = df(x1, x2)

      // the first output is x2, so it depends on x2 only, and the other way round
      x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
      x1_dx2 should approxEqual(Tensor2(x1.extent(Axis[A])).eye(x1.vtype))
      x2_dx1 should approxEqual(Tensor2(x2.extent(Axis[A])).eye(x2.vtype))
      x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

    it("d¹ of f: Tensor1[A] => Tensor1[B] keeps the output axis first"):
      def f(x: Tensor1[A, Float32]): Tensor1[B, Float32] = x.relabel(Axis[A] -> Axis[B]) *! 2.0f
      val df = Autodiff.jacFwd(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      df(x).axes shouldBe List("B", "A")
      df(x) should approxEqual((Tensor2(x.extent(Axis[A])).eye *! 2.0f).relabelAll((Axis[B], Axis[A])))

    it("d² of f(x1, x2) = sum(x1 * x2)"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] = (x1 * x2).sum
      val ddf = Autodiff.jacFwd(Autodiff.jacFwd(f.tupled))

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
      val ((x1_dx1, x1_dx2), (x2_dx1, x2_dx2)) = ddf(x1, x2)

      // d²/dx1² and d²/dx2² vanish, the mixed partials are the identity
      x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
      x1_dx2 should approxEqual(Tensor2(x1.extent(Axis[A])).eye(x1.vtype))
      x2_dx1 should approxEqual(Tensor2(x2.extent(Axis[A])).eye(x2.vtype))
      x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

  describe("hessian"):
    describe("single parameter function"):
      it("Hessian of f(x) = x^2"):
        def f(x: Tensor0[Float32]) = x * x
        val hf = Autodiff.hessian(f)

        val x = Tensor0(3.0f)
        hf(x) shouldEqual Tensor0(2.0f)

      it("Hessian of f(x) = sum(x^2)"):
        def f(x: Tensor1[A, Float32]) = (x * x).sum
        val hf = Autodiff.hessian(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 5.0f))
        hf(x) should approxEqual(Tensor2(x.extent(Axis[A])).eye(x.vtype) *! 2.0f)

      it("Hessian of f(x1, x2) = sum(x1 * x2)"):
        def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] = (x1 * x2).sum
        val hf = Autodiff.hessian(f.tupled)

        val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
        val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
        val (x1Grad, x2Grad) = hf(x1, x2)
        val (x1_dx1, x1_dx2) = x1Grad
        val (x2_dx1, x2_dx2) = x2Grad
        x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
        x1_dx2 should approxEqual(Tensor2(x1.extent(Axis[A])).eye(x1.vtype) *! Tensor0(1.0f))
        x2_dx1 should approxEqual(Tensor2(x2.extent(Axis[A])).eye(x2.vtype) *! Tensor0(1.0f))
        x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

  describe("jacobian of a function whose input and output axes differ"):

    it("non-square jacobian: Tensor1[A] => Tensor1[B]"):
      def f(x: Tensor1[A, Float32]): Tensor1[B, Float32] = x.relabel(Axis[A] -> Axis[B]) *! 2.0f
      val jf = Autodiff.jacobian(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      jf(x).axes shouldBe List("B", "A")
      jf(x) should approxEqual((Tensor2(x.extent(Axis[A])).eye *! 2.0f).relabelAll((Axis[B], Axis[A])))

    it("primes an input axis that collides with an output axis"):
      def f(x: Tensor2[A, B, Float32]): Tensor1[B, Float32] = x.sum(Axis[A])
      val jf = Autodiff.jacobian(f)

      val x = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1f)
      val jac = jf(x)
      jac.axes shouldBe List("B", "A", "B'")
      jac.shape(Axis[A]) shouldBe 3
      // d(sum over A)_b / dx(a, b') is 1 exactly when b == b', for every a
      jac.sum shouldEqual Tensor0(6.0f)

    it("primes a colliding input axis that is not the head of the output"):
      def f(x: Tensor1[B, Float32]): Tensor2[A, B, Float32] = x.broadcastTo(Shape(Axis[A] -> 3, Axis[B] -> 2))
      val jf = Autodiff.jacobian(f)

      val x = Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f))
      val jac = jf(x)
      jac.axes shouldBe List("A", "B", "B'")
      // the broadcast copies x, so d out(a, b) / dx(b') is 1 exactly when b == b'
      jac.sum shouldEqual Tensor0(6.0f)

    it("primes only the colliding axis of a multi-axis input"):
      def f(x: Tensor2[A, B, Float32]): Tensor2[C, A, Float32] =
        x.sum(Axis[B]).broadcastTo(Shape(Axis[C] -> 2, Axis[A] -> 3))
      val jf = Autodiff.jacobian(f)

      val x = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 4)).fill(1f)
      val jac = jf(x)
      jac.axes shouldBe List("C", "A", "A'", "B")
      jac.sum shouldEqual Tensor0(24.0f)

    it("jacobian over a tuple input with differing axes"):
      def f(x: Tensor1[A, Float32], y: Tensor1[B, Float32]): Tensor0[Float32] = x.sum * y.sum
      val jf = Autodiff.jacobian(f.tupled)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val y = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
      val (dx, dy) = jf(x, y)
      dx should approxEqual(Tensor1(Axis[A]).fromArray(Array(7.0f, 7.0f)))
      dy should approxEqual(Tensor1(Axis[B]).fromArray(Array(3.0f, 3.0f)))

    it("hessian of a scalar loss over two different axes"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[B, Float32]): Tensor0[Float32] = x1.sum * x2.sum
      val hf = Autodiff.hessian(f.tupled)

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val x2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
      val (d1, d2) = hf(x1, x2)
      val (d1_d1, d1_d2) = d1
      val (d2_d1, d2_d2) = d2
      d1_d1 should approxEqual(Tensor.like(d1_d1).fill(0f))
      d1_d2 should approxEqual(Tensor.like(d1_d2).fill(1f))
      d2_d1 should approxEqual(Tensor.like(d2_d1).fill(1f))
      d2_d2 should approxEqual(Tensor.like(d2_d2).fill(0f))

  describe("jacobian of structures that are not tensors or plain tuples"):

    val params = JacParams(
      Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f)),
      Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
    )

    it("differentiates a case class tree into a named tuple of its fields"):
      def f(p: JacParams): JacParams = p
      val jf = Autodiff.jacobian(f)
      val jac = jf(params)

      jac.w.w should approxEqual(Tensor2(params.w.extent(Axis[A])).eye)
      jac.w.b should approxEqual(Tensor.like(jac.w.b).fill(0f))
      jac.b.w should approxEqual(Tensor.like(jac.b.w).fill(0f))
      jac.b.b should approxEqual(Tensor2(params.b.extent(Axis[B])).eye)

    it("takes the hessian of a scalar loss over a case class tree"):
      def loss(p: JacParams): Tensor0[Float32] = (p.w * p.w).sum + (p.b * p.b).sum
      val hf = Autodiff.hessian(loss)
      val hess = hf(params)

      hess.w.w should approxEqual(Tensor2(params.w.extent(Axis[A])).eye *! 2.0f)
      hess.w.b should approxEqual(Tensor.like(hess.w.b).fill(0f))
      hess.b.b should approxEqual(Tensor2(params.b.extent(Axis[B])).eye *! 2.0f)

    it("differentiates a function returning a named tuple"):
      def f(x: Tensor1[A, Float32]): (u: Tensor1[A, Float32], v: Tensor1[A, Float32]) =
        (u = x *! 2.0f, v = x *! 3.0f)
      val jf = Autodiff.jacobian(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      val jac = jf(x)
      jac.u should approxEqual(Tensor2(x.extent(Axis[A])).eye *! 2.0f)
      jac.v should approxEqual(Tensor2(x.extent(Axis[A])).eye *! 3.0f)

  describe("Complex application"):
    it("case class support"):
      case class Params(w: Tensor1[A, Float32], b: Tensor0[Float32])
      def loss(data: Tensor1[A, Float32])(params: Params): Tensor0[Float32] =
        ((data * params.w).sum + params.b).pow(Tensor0(2.0f))
      val trainData = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val dloss = Autodiff.grad(loss(trainData))
      val params = Params(Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f)), Tensor0(3.0f))
      val dParams = dloss(params)
      dParams.value.w shouldEqual Tensor1(Axis[A]).fromArray(Array(16.0f, 32.0f))
