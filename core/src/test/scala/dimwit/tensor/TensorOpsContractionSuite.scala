package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsContractionSuite extends AnyFunSpec with Matchers:

  val v1 = Tensor1.fromArray(Axis[A], VType[Float])(
    Array(1.0f, 2.0f)
  )
  val v2 = Tensor1.fromArray(Axis[A], VType[Float])(
    Array(3.0f, 4.0f)
  )

  val m1 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
    Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
  )

  val m2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
    Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f))
  )

  describe("dot (Vectors)"):
    it("Tensor1[A] and Tensor1[A] (Standard Dot Product)"):
      v1.dot(Axis[A])(v2) shouldEqual Tensor0(11.0f)

  describe("dot (Matrices)"):
    it("Tensor2[A, B] and Tensor2[A, B] on axis A (row-wise)"):
      val res = m1.dot(Axis[A])(m2)

      res.shape.labels shouldBe List("B", "B'")
      res should approxEqual(
        Tensor.fromArray(res.shape, res.vtype)(
          Array(100.0f, 140.0f, 140.0f, 200.0f)
        )
      )

    it("Tensor2[A, B] and Tensor2[A, B] on axis B (column-wise)"):
      val res = m1.dot(Axis[B])(m2)

      res.shape.labels shouldBe List("A", "A'")
      res should approxEqual(
        Tensor.fromArray(res.shape, res.vtype)(
          Array(50.0f, 110.0f, 110.0f, 250.0f)
        )
      )

  describe("dot on different axis labels (A1 ~ A2)"):
    it("Tensor2[A, B] and Tensor2[C, D] using Axis mapping (A ~ C)"):
      val mCD = m2.relabelAll((Axis[C], Axis[D]))

      val res = m1.dot(Axis[A ~ C])(mCD)

      res.shape.labels shouldBe List("B", "D")
      res should approxEqual(
        Tensor.fromArray(res.shape, res.vtype)(
          Array(100.0f, 140.0f, 140.0f, 200.0f)
        )
      )

    it("~ should respect position-aware mapping in types"):
      val mCD = m2.relabelAll((Axis[C], Axis[D]))
      "m1.dot(Axis[A ~ C])(mCD)" should compile
      "m1.dot(Axis[C ~ A])(mCD)" shouldNot compile

  describe("outerProduct"):
    it("Tensor1[A] and Tensor1[B] to Tensor2[A, B]"):
      val vA = Tensor1.fromArray(Axis[A], VType[Float])(Array(1.0f, 2.0f))
      val vB = Tensor1.fromArray(Axis[B], VType[Float])(Array(10.0f, 20.0f))

      val res = vA.outerProduct(vB)
      res should approxEqual(
        Tensor.fromArray(res.shape, res.vtype)(
          Array(10.0f, 20.0f, 20.0f, 40.0f)
        )
      )
