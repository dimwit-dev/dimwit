package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given

class TensorOpsElementwiseSuite extends DimwitTest:

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

    it("arcsin/arccos/arctan"):
      Tensor.like(t2).fill(0.5f).arcsin should approxEqual(Tensor.like(t2).fill((math.Pi / 6).toFloat), tolerance = 1e-5f)
      Tensor.like(t2).fill(0.5f).arccos should approxEqual(Tensor.like(t2).fill((math.Pi / 3).toFloat), tolerance = 1e-5f)
      Tensor.like(t2).fill(1.0f).arctan should approxEqual(Tensor.like(t2).fill((math.Pi / 4).toFloat), tolerance = 1e-5f)

    it("floor/ceil/round"):
      val t = Tensor.like(t2).fromArray(Array(-1.5f, 0.4f, 1.5f, 2.6f))
      t.floor should approxEqual(Tensor.like(t2).fromArray(Array(-2.0f, 0.0f, 1.0f, 2.0f)))
      t.ceil should approxEqual(Tensor.like(t2).fromArray(Array(-1.0f, 1.0f, 2.0f, 3.0f)))
      t.round should approxEqual(Tensor.like(t2).fromArray(Array(-2.0f, 0.0f, 2.0f, 3.0f)))

    it("isnan/isfinite"):
      val t = Tensor.like(t2).fromArray(Array(Float.NaN, Float.PositiveInfinity, 1.0f, 0.0f))
      t.isnan shouldEqual Tensor.like(b2).fromArray(Array(true, false, false, false))
      t.isfinite shouldEqual Tensor.like(b2).fromArray(Array(false, false, true, true))

    it("mod"):
      val t = Tensor.like(t2).fromArray(Array(-7.0f, 7.0f, -7.0f, 7.0f))
      val divisor = Tensor.like(t2).fromArray(Array(3.0f, 3.0f, -3.0f, -3.0f))
      (t % divisor) should approxEqual(Tensor.like(t2).fromArray(Array(2.0f, 1.0f, -1.0f, -2.0f)))

    it("mod broadcasting (%!)"):
      val t = Tensor1(Axis[A]).fromArray(Array(-7.0f, 7.0f))
      (t %! Tensor0(3.0f)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(2.0f, 1.0f)))

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

    it("mod"):
      val t = Tensor.like(i2).fromArray(Array(-7, 7, -7, 7))
      val divisor = Tensor.like(i2).fromArray(Array(3, 3, -3, -3))
      (t % divisor) shouldEqual Tensor.like(i2).fromArray(Array(2, 1, -1, -2))

    it("clip"):
      i2.clip(0, 1) shouldEqual Tensor.like(i2).fromArray(Array(0, 0, 1, 1))

    it("unary_-"):
      (-i2) shouldEqual Tensor.like(i2).fromArray(Array(1, 0, -1, -2))

  describe("Boolean ops (Tensor2)"):

    val c2 = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(
        Array(true, true),
        Array(false, false)
      )
    )

    it("inverse (!)"):
      (!b2) shouldEqual Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(false, true), Array(true, false))
      )

    it("and"):
      val expected = Tensor.like(b2).fromArray(Array(true, false, false, false))
      (b2 and c2) shouldEqual expected
      (b2 and c2) shouldEqual (c2 and b2)

    it("or"):
      val expected = Tensor.like(b2).fromArray(Array(true, true, false, true))
      (b2 or c2) shouldEqual expected
      (b2 or c2) shouldEqual (c2 or b2)

    it("xor"):
      val expected = Tensor.like(b2).fromArray(Array(false, true, false, true))
      (b2 xor c2) shouldEqual expected
      (b2 xor c2) shouldEqual (c2 xor b2)

    it("identities"):
      val allTrue = Tensor.like(b2).fill(true)
      val allFalse = Tensor.like(b2).fill(false)
      (b2 and allTrue) shouldEqual b2
      (b2 and allFalse) shouldEqual allFalse
      (b2 or allFalse) shouldEqual b2
      (b2 or allTrue) shouldEqual allTrue
      (b2 xor allFalse) shouldEqual b2
      (b2 xor allTrue) shouldEqual !b2
      (b2 xor b2) shouldEqual allFalse
      // De Morgan
      (!(b2 and c2)) shouldEqual ((!b2) or (!c2))
      (!(b2 or c2)) shouldEqual ((!b2) and (!c2))

    it("broadcasting (and_! / or_! / xor_!)"):
      val bA = Tensor1(Axis[A]).fromArray(Array(true, false))
      (b2 and_! bA) shouldEqual Tensor.like(b2).fromArray(Array(true, false, false, false))
      (b2 or_! bA) shouldEqual Tensor.like(b2).fromArray(Array(true, true, false, true))
      (b2 xor_! bA) shouldEqual Tensor.like(b2).fromArray(Array(false, true, false, true))
      (bA and_! b2) shouldEqual (b2 and_! bA)

  describe("Casting Ops (Tensor2)"):

    it("boolean casting"):
      b2.asBool shouldEqual b2
      b2.asInt32 shouldEqual Tensor(b2.shape).fromArray(Array(1, 0, 0, 1))
      b2.asFloat32 should approxEqual(Tensor(b2.shape).fromArray(Array(1.0f, 0.0f, 0.0f, 1.0f)))

    it("int casting"):
      i2.asBool shouldEqual Tensor(i2.shape).fromArray(Array(true, false, true, true))
      i2.asInt32 shouldEqual i2
      i2.asFloat32 should approxEqual(Tensor(i2.shape).fromArray(Array(-1.0f, 0.0f, 1.0f, 2.0f)))

    it("float casting"):
      val f2 = Tensor.like(t2).fromArray(Array(-1.1f, 0.0f, 0.9f, 2.5f))
      f2.asBool shouldEqual Tensor(f2.shape).fromArray(Array(true, false, true, true))
      f2.asInt32 shouldEqual Tensor(f2.shape).fromArray(Array(-1, 0, 0, 2))
      f2.asFloat32 shouldEqual f2
