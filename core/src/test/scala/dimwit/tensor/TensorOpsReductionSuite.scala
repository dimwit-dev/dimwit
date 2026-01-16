package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsReductionSuite extends AnyFunSpec with Matchers:

  val t2 = Tensor2(
    Axis[A],
    Axis[B]
  ).fromArray(
    Array(
      Array(1.0f, 2.0f, 3.0f),
      Array(4.0f, 5.0f, 6.0f)
    )
  )

  val b2 = Tensor2(
    Axis[A],
    Axis[B]
  ).fromArray(
    Array(
      Array(true, true),
      Array(true, false)
    )
  )

  describe("Equality Ops"):

    it("== (Scala Boolean)"):
      (t2 == t2) shouldBe true
      (t2 == (t2 *! Tensor0(2.0f))) shouldBe false

    it("=== (Tensor0[Boolean])"):
      (t2 === t2).item shouldBe true
      (t2 === (t2 *! Tensor0(0.0f))).item shouldBe false

  describe("Reduction Ops"):
    it("sum"):
      t2.sum shouldEqual Tensor0(21.0f)

    it("sum axis A"):
      val res = t2.sum(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(5.0f, 7.0f, 9.0f)))

    it("sum axis B"):
      val res = t2.sum(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(6.0f, 15.0f)))
    it("mean"):
      t2.mean shouldEqual Tensor0(3.5f)

    it("mean axis A"):
      val res = t2.mean(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(2.5f, 3.5f, 4.5f)))

    it("mean axis B"):
      val res = t2.mean(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(2.0f, 5.0f)))

    it("std"):
      t2.std.item should be(1.7078f +- 0.001f)

    it("std axis A"):
      val res = t2.std(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(1.5f, 1.5f, 1.5f)))

    it("std axis B"):
      val res = t2.std(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(0.8164966f, 0.8164966f)))

    it("quantile"):
      t2.quantile(0.5f) shouldEqual Tensor0(3.5f)

    it("quantile axis A"):
      val res = t2.quantile(0.25f, axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(1.75f, 2.75f, 3.75f)))

    it("quantile axis B"):
      val res = t2.quantile(0.25f, axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(1.5f, 4.5f)))

    it("median"):
      t2.median shouldEqual Tensor0(3.5f)

    it("median axis A"):
      val res = t2.median(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(2.5f, 3.5f, 4.5f)))

    it("median axis B"):
      val res = t2.median(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(2.0f, 5.0f)))

    it("max"):
      t2.max shouldEqual Tensor0(6.0f)

    it("max axis A"):
      val res = t2.max(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(4.0f, 5.0f, 6.0f)))

    it("max axis B"):
      val res = t2.max(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(3.0f, 6.0f)))

    it("min"):
      t2.min shouldEqual Tensor0(1.0f)

    it("min axis A"):
      val res = t2.min(axis = Axis[A])
      res should approxEqual(Tensor.like(res).fromArray(Array(1.0f, 2.0f, 3.0f)))

    it("min axis B"):
      val res = t2.min(axis = Axis[B])
      res should approxEqual(Tensor.like(res).fromArray(Array(1.0f, 4.0f)))

    it("argmax"):
      t2.argmax shouldEqual Tensor0(5)

    it("argmax axis A"):
      val res = t2.argmax(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(1, 1, 1))

    it("argmax axis B"):
      val res = t2.argmax(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(2, 2))

    it("argmin"):
      t2.argmin shouldEqual Tensor0(0)

    it("argmin axis A"):
      val res = t2.argmin(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(0, 0, 0))

    it("argmin axis B"):
      val res = t2.argmin(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(0, 0))

  describe("Boolean Reductions"):
    it("all"):
      b2.all shouldEqual Tensor0(false)
      val allTrue = Tensor.like(b2).fill(true)
      allTrue.all shouldEqual Tensor0(true)

    it("any"):
      b2.any shouldEqual Tensor0(true)
      val allFalse = Tensor.like(b2).fill(false)
      allFalse.any shouldEqual Tensor0(false)

  describe("Approximate Equality") {
    it("approxEquals"):
      val t2Near = t2 *! Tensor0(1.0000001f)
      t2.approxEquals(t2Near).item shouldBe true
      val t2Far = t2 *! Tensor0(1.1f)
      t2.approxEquals(t2Far).item shouldBe false
  }
