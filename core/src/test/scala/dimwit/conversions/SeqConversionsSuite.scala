package dimwit

import dimwit.*
import dimwit.conversions.*
import scala.collection.immutable.ArraySeq
import scala.compiletime.testing.typeCheckErrors

class SeqConversionsSuite extends DimwitTest:

  def withJaxX64Support[R](block: => R): R =
    import me.shadaj.scalapy.py
    val jaxConfig = py.module("jax").config
    val current = jaxConfig.jax_enable_x64.as[Boolean]
    jaxConfig.update("jax_enable_x64", true)
    val res = block
    jaxConfig.update("jax_enable_x64", current)
    res

  describe("toTensor1"):

    it("derives the value type from the element type"):
      Seq(true, false).toTensor1(Axis[A]).dtype shouldBe DType.Bool
      Seq(1.toByte, 2.toByte).toTensor1(Axis[A]).dtype shouldBe DType.Int8
      Seq(1.toShort, 2.toShort).toTensor1(Axis[A]).dtype shouldBe DType.Int16
      Seq(1, 2).toTensor1(Axis[A]).dtype shouldBe DType.Int32
      Seq(1.0f, 2.0f).toTensor1(Axis[A]).dtype shouldBe DType.Float32
      withJaxX64Support:
        Seq(1L, 2L).toTensor1(Axis[A]).dtype shouldBe DType.Int64
        Seq(1.0, 2.0).toTensor1(Axis[A]).dtype shouldBe DType.Float64

    it("keeps the values and their order"):
      val t: Tensor1[A, Float32] = Seq(3.0f, 1.0f, 2.0f).toTensor1(Axis[A])
      t.shape shouldBe Shape1(Axis[A] -> 3)
      t.toArray shouldBe Array(3.0f, 1.0f, 2.0f)

    it("accepts any Seq"):
      List(1, 2, 3).toTensor1(Axis[A]).toArray shouldBe Array(1, 2, 3)
      Vector(1, 2, 3).toTensor1(Axis[A]).toArray shouldBe Array(1, 2, 3)
      ArraySeq(1, 2, 3).toTensor1(Axis[A]).toArray shouldBe Array(1, 2, 3)
      (1 to 3).toTensor1(Axis[A]).toArray shouldBe Array(1, 2, 3)
      (1 to 3).map(_ * 0.5f).toTensor1(Axis[A]).toArray shouldBe Array(0.5f, 1.0f, 1.5f)

    it("takes the value type explicitly"):
      Seq(1.0f, 2.0f).toTensor1(Axis[A], VType[Float16]).dtype shouldBe DType.Float16
      Seq(1, 2).toTensor1(Axis[A], VType[Int8]).dtype shouldBe DType.Int8
      Seq(true).toTensor1(Axis[A], VType[Bool]).dtype shouldBe DType.Bool

    it("converts an empty Seq to an axis of extent 0"):
      Seq.empty[Float].toTensor1(Axis[A]).shape shouldBe Shape1(Axis[A] -> 0)

  describe("toTensor2"):

    it("derives the value type from the element type"):
      Seq(Seq(true), Seq(false)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Bool
      Seq(Seq(1.toByte), Seq(2.toByte)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Int8
      Seq(Seq(1.toShort), Seq(2.toShort)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Int16
      Seq(Seq(1), Seq(2)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Int32
      Seq(Seq(1.0f), Seq(2.0f)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Float32
      withJaxX64Support:
        Seq(Seq(1L), Seq(2L)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Int64
        Seq(Seq(1.0), Seq(2.0)).toTensor2(Axis[A], Axis[B]).dtype shouldBe DType.Float64

    it("uses the inner Seqs as rows"):
      val t: Tensor2[A, B, Float32] = Seq(Seq(1.0f, 2.0f, 3.0f), Seq(4.0f, 5.0f, 6.0f)).toTensor2(Axis[A], Axis[B])
      t.shape shouldBe Shape2(Axis[A] -> 2, Axis[B] -> 3)
      t.toArray shouldBe Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f))

    it("accepts any Seq of Seqs"):
      List(Vector(1, 2), Vector(3, 4)).toTensor2(Axis[A], Axis[B]).toArray shouldBe Array(Array(1, 2), Array(3, 4))
      (0 until 2).map(i => (0 until 3).map(j => 10 * i + j)).toTensor2(Axis[A], Axis[B]).toArray shouldBe
        Array(Array(0, 1, 2), Array(10, 11, 12))

    it("takes the value type explicitly"):
      Seq(Seq(1.0f, 2.0f)).toTensor2(Axis[A], Axis[B], VType[Float16]).dtype shouldBe DType.Float16
      Seq(Seq(1, 2)).toTensor2(Axis[A], Axis[B], VType[Int8]).dtype shouldBe DType.Int8

    it("allows rows of length 0"):
      Seq(Seq.empty[Float], Seq.empty[Float]).toTensor2(Axis[A], Axis[B]).shape shouldBe Shape2(Axis[A] -> 2, Axis[B] -> 0)

    it("rejects rows of different lengths"):
      val error = intercept[IllegalArgumentException]:
        Seq(Seq(1.0f, 2.0f), Seq(3.0f), Seq(4.0f, 5.0f, 6.0f)).toTensor2(Axis[A], Axis[B])
      error.getMessage should include("row 0 has length 2 and row 1 has length 1")

    it("rejects a Seq without rows"):
      val error = intercept[IllegalArgumentException]:
        Seq.empty[Seq[Float]].toTensor2(Axis[A], Axis[B])
      error.getMessage should include("extent of axis B is unknown")

  describe("Tensor2 fromArray"):
    it("rejects ragged rows even if the total number of values fits"):
      // 3 rows, the first of length 2, and 6 values in total: used to become a wrong 3x2 matrix
      intercept[IllegalArgumentException]:
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f), Array(4.0f, 5.0f, 6.0f)))

  describe("compile errors"):
    it("does not compile for unsupported element types"):
      typeCheckErrors("""Seq("a", "b").toTensor1(Axis[A])""") should not be empty
      typeCheckErrors("""Seq(Seq("a")).toTensor2(Axis[A], Axis[B])""") should not be empty

    it("does not compile for a value type that does not fit the elements"):
      typeCheckErrors("Seq(1.0f, 2.0f).toTensor1(Axis[A], VType[Int32])") should not be empty
      typeCheckErrors("Seq(1, 2).toTensor1(Axis[A], VType[Float32])") should not be empty
      typeCheckErrors("Seq(Seq(1.0f)).toTensor2(Axis[A], Axis[B], VType[Bool])") should not be empty

    it("does not compile without a label for the axis"):
      typeCheckErrors("Seq(1.0f).toTensor1(Axis[String])") should not be empty
