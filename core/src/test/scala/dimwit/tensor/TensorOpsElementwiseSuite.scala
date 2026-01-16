package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsElementwiseSuite extends AnyFunSpec with Matchers:

  val t2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(
      Array(-1.0f, 0.0f),
      Array(1.0f, 4.0f)
    )
  )

  val i2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(
      Array(-1, 0),
      Array(1, 2)
    )
  )

  val b2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(
      Array(true, false),
      Array(false, true)
    )
  )

  describe("Float ops (Tensor2)"):

    it("abs"):
      t2.abs should approxEqual(Tensor.like(t2).fromArray(Array(1.0f, 0.0f, 1.0f, 4.0f)))

    it("sign"):
      t2.sign should approxEqual(Tensor.like(t2).fromArray(Array(-1.0f, 0.0f, 1.0f, 1.0f)))

    it("pow"):
      t2.pow(Tensor0(2.0f)) should approxEqual(Tensor.like(t2).fromArray(Array(1.0f, 0.0f, 1.0f, 16.0f)))

    it("sqrt"):
      val tPos = Tensor.like(t2).fromArray(Array(4.0f, 9.0f, 16.0f, 25.0f))
      tPos.sqrt should approxEqual(Tensor.like(t2).fromArray(Array(2.0f, 3.0f, 4.0f, 5.0f)))

    it("exp/log (identity)"):
      val tZero = Tensor.like(t2).fill(0f)
      val tOne = Tensor.like(t2).fill(1f)
      tZero.exp should approxEqual(tOne)
      tOne.log should approxEqual(tZero)

    it("sin/cos/tanh"):
      val tZero = Tensor.like(t2).fill(0f)
      tZero.sin should approxEqual(tZero)
      tZero.cos should approxEqual(Tensor.like(t2).fill(1f))
      tZero.tanh should approxEqual(tZero)

    it("clip"):
      t2.clip(0.0f, 2.0f) should approxEqual(Tensor.like(t2).fromArray(Array(0.0f, 0.0f, 1.0f, 2.0f)))

    it("unary_-"):
      (-t2) should approxEqual(Tensor.like(t2).fromArray(Array(1.0f, 0.0f, -1.0f, -4.0f)))

    it("approxEquals / approxElementEquals"):
      val t2Near = t2 *! Tensor0(1.0000001f)
      t2.approxEquals(t2Near).item shouldBe true
      t2.approxElementEquals(t2Near).all.item shouldBe true

  describe("Int ops (Tensor2)"):

    it("abs"):
      i2.abs shouldEqual Tensor.like(i2).fromArray(Array(1, 0, 1, 2))

    it("sign"):
      i2.sign shouldEqual Tensor.like(i2).fromArray(Array(-1, 0, 1, 1))

    it("pow"):
      i2.pow(Tensor0(3)) shouldEqual Tensor.like(i2).fromArray(Array(-1, 0, 1, 8))

    it("clip"):
      i2.clip(0, 1) shouldEqual Tensor.like(i2).fromArray(Array(0, 0, 1, 1))

    it("unary_-"):
      (-i2) shouldEqual Tensor.like(i2).fromArray(Array(1, 0, -1, -2))

  describe("Boolean ops (Tensor2)"):

    it("inverse (!)"):
      (!b2) shouldEqual Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(false, true), Array(true, false))
      )

  describe("Casting Ops (Tensor2)"):

    it("boolean casting"):
      b2.asBoolean shouldEqual b2
      b2.asInt shouldEqual Tensor(b2.shape).fromArray(Array(1, 0, 0, 1))
      b2.asFloat should approxEqual(Tensor(b2.shape).fromArray(Array(1.0f, 0.0f, 0.0f, 1.0f)))

    it("int casting"):
      i2.asBoolean shouldEqual Tensor(i2.shape).fromArray(Array(true, false, true, true))
      i2.asInt shouldEqual i2
      i2.asFloat should approxEqual(Tensor(i2.shape).fromArray(Array(-1.0f, 0.0f, 1.0f, 2.0f)))

    it("float casting"):
      val f2 = Tensor.like(t2).fromArray(Array(-1.1f, 0.0f, 0.9f, 2.5f))
      f2.asBoolean shouldEqual Tensor(f2.shape).fromArray(Array(true, false, true, true))
      f2.asInt shouldEqual Tensor(f2.shape).fromArray(Array(-1, 0, 0, 2))
      f2.asFloat shouldEqual f2
