package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsBroadcastSuite extends AnyFunSpec with Matchers:

  val tA = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 2.0f))

  val tAB = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

  val tAB2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(100.0f, 200.0f)))

  val iA = Tensor1.fromArray(Axis[A], VType[Int])(Array(1, 2))
  val iAB = Tensor2.fromArray(Axis[A], Axis[B], VType[Int])(Array(Array(1, 2), Array(3, 4)))

  describe("Scalar Broadcasting"):

    describe("Int"):
      it("Addition"):
        (5 +! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(6, 7, 8, 9))
        (5 +! iAB) shouldEqual (iAB +! 5)

      it("Subtraction"):
        (5 -! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(4, 3, 2, 1))
        (iAB -! 5) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(-4, -3, -2, -1))

      it("Multiplication"):
        (3 *! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(3, 6, 9, 12))
        (3 *! iAB) shouldEqual (iAB *! 3)

      it("No Int Division Supported"):
        "5 /! iAB" shouldNot compile
        "iAB /! 5" shouldNot compile

    describe("Float"):

      it("Addition"):
        (2.0f +! tAB) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(12.0f, 22.0f, 32.0f, 42.0f))
        (2.0f +! tAB) shouldEqual (tAB +! 2.0f)

      it("Subtraction"):
        (5.0f -! tAB) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(-5.0f, -15.0f, -25.0f, -35.0f))
        (tAB -! 5.0f) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(5.0f, 15.0f, 25.0f, 35.0f))

      it("Multiplication"):
        (2.0f *! tAB) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(20.0f, 40.0f, 60.0f, 80.0f))
        (2.0f *! tAB) shouldEqual (tAB *! 2.0f)

      it("Division"):
        (2.0f /! tAB) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(0.2f, 0.1f, 0.06666667f, 0.05f))
        (tAB /! 2.0f) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(Array(5.0f, 10.0f, 15.0f, 20.0f))

  describe("Vector-to-Tensor Broadcasting"):

    describe("Int"):

      it("Addition"):
        (iA +! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(2, 3, 5, 6))
        (iA +! iAB) shouldEqual (iAB +! iA)

      it("Subtraction"):
        (iA -! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(0, -1, -1, -2))
        (iAB -! iA) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(0, 1, 1, 2))

      it("Multiplication"):
        (iA *! iAB) shouldEqual Tensor.fromArray(iAB.shape, iAB.vtype)(Array(1, 2, 6, 8))
        (iA *! iAB) shouldEqual (iAB *! iA)

      it("No Int Division Supported"):
        "iA /! iAB" shouldNot compile
        "iAB /! iA" shouldNot compile

    describe("Float"):

      it("Addition"):
        (tAB +! tA) should approxEqual(
          Tensor.fromArray(tAB.shape, tAB.vtype)(
            Array(11.0f, 21.0f, 32.0f, 42.0f)
          )
        )
        (tAB +! tA) shouldEqual (tA +! tAB)

      it("Subtraction"):
        (tAB -! tA) should approxEqual(
          Tensor.fromArray(tAB.shape, tAB.vtype)(
            Array(9.0f, 19.0f, 28.0f, 38.0f)
          )
        )
        (tA -! tAB) shouldEqual Tensor.fromArray(tAB.shape, tAB.vtype)(
          Array(-9.0f, -19.0f, -28.0f, -38.0f)
        )

      it("Multiplication"):
        (tAB *! tA) should approxEqual(
          Tensor.fromArray(tAB.shape, tAB.vtype)(
            Array(10.0f, 20.0f, 60.0f, 80.0f)
          )
        )
        (tAB *! tA) should approxEqual(tA *! tAB)

      it("Division"):
        (tAB /! tA) should approxEqual(
          Tensor.fromArray(tAB.shape, tAB.vtype)(
            Array(10.0f, 20.0f, 15.0f, 20.0f)
          )
        )
        (tA /! tAB) should approxEqual(
          Tensor.fromArray(tAB.shape, tAB.vtype)(
            Array(0.1f, 0.05f, 0.06666667f, 0.05f)
          )
        )

  describe("Tensor-to-Tensor Broadcasting (complex)"):

    val tABCD = Tensor.fromArray(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2), VType[Float])(
      Array.range(1, 17).map(_.toFloat)
    )

    it("AB broadcastTo ABCD"):
      val AB = Tensor.fromArray(Shape(Axis[A] -> 2, Axis[B] -> 2), VType[Float])(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = AB.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[C] -> 0, Axis[D] -> 0)) should approxEqual(AB)
      res.slice((Axis[C] -> 1, Axis[D] -> 0)) should approxEqual(AB)
      res.slice((Axis[C] -> 0, Axis[D] -> 1)) should approxEqual(AB)
      res.slice((Axis[C] -> 1, Axis[D] -> 1)) should approxEqual(AB)

    it("BC broadcastTo ABCD"):
      val BC = Tensor.fromArray(Shape(Axis[B] -> 2, Axis[C] -> 2), VType[Float])(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = BC.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A] -> 0, Axis[D] -> 0)) should approxEqual(BC)
      res.slice((Axis[A] -> 1, Axis[D] -> 0)) should approxEqual(BC)
      res.slice((Axis[A] -> 0, Axis[D] -> 1)) should approxEqual(BC)
      res.slice((Axis[A] -> 1, Axis[D] -> 1)) should approxEqual(BC)

    it("CD broadcastTo ABCD"):
      val CD = Tensor.fromArray(Shape(Axis[C] -> 2, Axis[D] -> 2), VType[Float])(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = CD.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A] -> 0, Axis[B] -> 0)) should approxEqual(CD)
      res.slice((Axis[A] -> 1, Axis[B] -> 0)) should approxEqual(CD)
      res.slice((Axis[A] -> 0, Axis[B] -> 1)) should approxEqual(CD)
      res.slice((Axis[A] -> 1, Axis[B] -> 1)) should approxEqual(CD)

  describe("Disallow"):

    val tABCD = Tensor.fromArray(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2), VType[Float])(
      Array.range(1, 17).map(_.toFloat)
    )

    it("Broadcasting same tensor"):
      "tAB +! tAB" shouldNot compile
      "tAB + tAB" should compile

    it("Shape broadcasting"):
      // JAX allows this, but we disallow it as it often hides bugs
      // dimwit broadcasting only adds missing axes, never changes shapes of existing axes
      val tAB1 = tA.appendAxis(Axis[B])
      an[IllegalArgumentException] should be thrownBy (tAB1 +! tABCD)

  describe("Operator Precedence"):

    it("multiplication (*!) binds tighter than addition (+!)"):
      val tA = Tensor.ones(Shape1(tAB.shape.dim(Axis[A])), VType[Float])
      val res = tAB *! Tensor0(2.0f) +! tA
      val correct = (tAB *! Tensor0(2.0f)) +! tA
      val wrong = tAB *! (Tensor0(2.0f) +! tA)
      res should approxEqual(correct)
      res shouldNot approxEqual(wrong)

  describe("Mixed Broadcasting Cases"):

    it("Broadcasting ab + bc to abc"):
      val ab = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(1.0f, 2.0f)))
      val bc = Tensor2.fromArray(Axis[B], Axis[C], VType[Float])(Array(Array(10.0f), Array(20.0f)))
      "ab +! bc" shouldNot compile // TODO add support for this
