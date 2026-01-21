package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsBroadcastSuite extends AnyFunSpec with Matchers:

  val tA = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))

  val tAB = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

  val tAB2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(100.0f, 200.0f)))
  val iA = Tensor1(Axis[A]).fromArray(Array(1, 2))
  val iAB = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1, 2), Array(3, 4)))

  describe("Scalar Broadcasting"):

    describe("Int"):
      it("Addition"):
        (5 +! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(6, 7, 8, 9))
        (5 +! iAB) shouldEqual (iAB +! 5)

      it("Subtraction"):
        (5 -! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(4, 3, 2, 1))
        (iAB -! 5) shouldEqual Tensor.like(iAB).fromArray(Array(-4, -3, -2, -1))

      it("Multiplication"):
        (3 *! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(3, 6, 9, 12))
        (3 *! iAB) shouldEqual (iAB *! 3)

      it("No Int Division Supported"):
        "5 /! iAB" shouldNot compile
        "iAB /! 5" shouldNot compile

    describe("Float"):

      it("Addition"):
        (2.0f +! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(12.0f, 22.0f, 32.0f, 42.0f))
        (2.0f +! tAB) shouldEqual (tAB +! 2.0f)

      it("Subtraction"):
        (5.0f -! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(-5.0f, -15.0f, -25.0f, -35.0f))
        (tAB -! 5.0f) shouldEqual Tensor.like(tAB).fromArray(Array(5.0f, 15.0f, 25.0f, 35.0f))

      it("Multiplication"):
        (2.0f *! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(20.0f, 40.0f, 60.0f, 80.0f))
        (2.0f *! tAB) shouldEqual (tAB *! 2.0f)

      it("Division"):
        (2.0f /! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(0.2f, 0.1f, 0.06666667f, 0.05f))
        (tAB /! 2.0f) shouldEqual Tensor.like(tAB).fromArray(Array(5.0f, 10.0f, 15.0f, 20.0f))

  describe("Vector-to-Tensor Broadcasting"):

    describe("Int"):

      it("Addition"):
        (iA +! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(2, 3, 5, 6))
        (iA +! iAB) shouldEqual (iAB +! iA)

      it("Subtraction"):
        (iA -! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(0, -1, -1, -2))
        (iAB -! iA) shouldEqual Tensor.like(iAB).fromArray(Array(0, 1, 1, 2))

      it("Multiplication"):
        (iA *! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(1, 2, 6, 8))
        (iA *! iAB) shouldEqual (iAB *! iA)

      it("No Int Division Supported"):
        "iA /! iAB" shouldNot compile
        "iAB /! iA" shouldNot compile

    describe("Float"):

      it("Addition"):
        (tAB +! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(11.0f, 21.0f, 32.0f, 42.0f)
          )
        )
        (tAB +! tA) shouldEqual (tA +! tAB)

      it("Subtraction"):
        (tAB -! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(9.0f, 19.0f, 28.0f, 38.0f)
          )
        )
        (tA -! tAB) shouldEqual Tensor.like(tAB).fromArray(
          Array(-9.0f, -19.0f, -28.0f, -38.0f)
        )

      it("Multiplication"):
        (tAB *! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(10.0f, 20.0f, 60.0f, 80.0f)
          )
        )
        (tAB *! tA) should approxEqual(tA *! tAB)

      it("Division"):
        (tAB /! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(10.0f, 20.0f, 15.0f, 20.0f)
          )
        )
        (tA /! tAB) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(0.1f, 0.05f, 0.06666667f, 0.05f)
          )
        )

  describe("Tensor-to-Tensor Broadcasting (complex)"):

    val tABCD = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2)).fromArray(
      Array.range(1, 17).map(_.toFloat)
    )

    it("AB broadcastTo ABCD"):
      val AB = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = AB.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[C].at(0), Axis[D].at(0))) should approxEqual(AB)
      res.slice((Axis[C].at(1), Axis[D].at(0))) should approxEqual(AB)
      res.slice((Axis[C].at(0), Axis[D].at(1))) should approxEqual(AB)
      res.slice((Axis[C].at(1), Axis[D].at(1))) should approxEqual(AB)

    it("BC broadcastTo ABCD"):
      val BC = Tensor(Shape(Axis[B] -> 2, Axis[C] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = BC.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A].at(0), Axis[D].at(0))) should approxEqual(BC)
      res.slice((Axis[A].at(1), Axis[D].at(0))) should approxEqual(BC)
      res.slice((Axis[A].at(0), Axis[D].at(1))) should approxEqual(BC)
      res.slice((Axis[A].at(1), Axis[D].at(1))) should approxEqual(BC)

    it("CD broadcastTo ABCD"):
      val CD = Tensor(Shape(Axis[C] -> 2, Axis[D] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = CD.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A].at(0), Axis[B].at(0))) should approxEqual(CD)
      res.slice((Axis[A].at(1), Axis[B].at(0))) should approxEqual(CD)
      res.slice((Axis[A].at(0), Axis[B].at(1))) should approxEqual(CD)
      res.slice((Axis[A].at(1), Axis[B].at(1))) should approxEqual(CD)

  describe("Disallow"):

    val tABCD = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2)).fromArray(
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
      val tA = Tensor(Shape1(tAB.shape.extent(Axis[A]))).fill(1f)
      val res = tAB *! Tensor0(2.0f) +! tA
      val correct = (tAB *! Tensor0(2.0f)) +! tA
      val wrong = tAB *! (Tensor0(2.0f) +! tA)
      res should approxEqual(correct)
      res shouldNot approxEqual(wrong)

  describe("Mixed Broadcasting Cases"):

    it("Broadcasting ab + bc to abc"):
      val ab = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val bc = Tensor2(Axis[B], Axis[C]).fromArray(Array(Array(10.0f), Array(20.0f)))
      "ab +! bc" shouldNot compile // TODO add support for this
