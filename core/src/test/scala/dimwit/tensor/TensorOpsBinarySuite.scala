package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsBinarySuite extends AnyFunSpec with Matchers:

  val t2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f))
  )
  val t2_2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(2.0f, 4.0f), Array(5.0f, 8.0f))
  )

  val i2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(10, 20), Array(30, 40))
  )
  val i2_2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(2, 4), Array(5, 8))
  )

  describe("Float Binary Ops"):
    it("Addition (+)"):
      (t2 + t2_2) shouldEqual Tensor.like(t2).fromArray(Array(12.0f, 24.0f, 35.0f, 48.0f))

    it("Subtraction (-)"):
      (t2 - t2_2) shouldEqual Tensor.like(t2).fromArray(Array(8.0f, 16.0f, 25.0f, 32.0f))

    it("Multiplication (*)"):
      (t2 * t2_2) shouldEqual Tensor.like(t2).fromArray(Array(20.0f, 80.0f, 150.0f, 320.0f))

    it("Division (/)"):
      (t2 / t2_2) shouldEqual Tensor.like(t2).fromArray(Array(5.0f, 5.0f, 6.0f, 5.0f))

    it("Comparisons (<, <=, >, >=)"):
      (t2 < t2_2).asBoolean shouldEqual Tensor(t2.shape).fromArray(Array(false, false, false, false))
      (t2 > t2_2).asBoolean shouldEqual Tensor(t2.shape).fromArray(Array(true, true, true, true))

    it("elementEquals"):
      (t2 `elementEquals` t2) shouldEqual Tensor(t2.shape).fromArray(Array(true, true, true, true))
      (t2 `elementEquals` t2_2) shouldEqual Tensor(t2.shape).fromArray(Array(false, false, false, false))

  describe("Int Binary Ops"):
    it("Addition (+)"):
      (i2 + i2_2) shouldEqual Tensor.like(i2).fromArray(Array(12, 24, 35, 48))

    it("Subtraction (-)"):
      (i2 - i2_2) shouldEqual Tensor.like(i2).fromArray(Array(8, 16, 25, 32))

    it("Multiplication (*)"):
      (i2 * i2_2) shouldEqual Tensor.like(i2).fromArray(Array(20, 80, 150, 320))

    it("Comparisons (<, <=, >, >=)"):
      (i2 < i2_2).asBoolean shouldEqual Tensor(i2.shape).fromArray(Array(false, false, false, false))
      (i2 >= i2_2).asBoolean shouldEqual Tensor(i2.shape).fromArray(Array(true, true, true, true))
