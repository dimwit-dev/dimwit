package dimwit.tensor

import dimwit.*
import scala.compiletime.testing.typeCheckErrors

class TensorCreationSuite extends DimwitTest:

  def withJaxX64Support[R](block: => R): R =
    import me.shadaj.scalapy.py
    val jaxConfig = py.module("jax").config
    val current = jaxConfig.jax_enable_x64.as[Boolean]
    jaxConfig.update("jax_enable_x64", true)
    val res = block
    jaxConfig.update("jax_enable_x64", current)
    res

  describe("Default settings"):
    describe("Tensor fill"):
      it("Fill tensors with tensor types"):
        val intTensor = Tensor(Shape2(Axis[A] -> 4, Axis[B] -> 5)).fill(42)
        intTensor.dtype shouldBe DType.Int32
        val floatTensor = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(3.14f)
        floatTensor.dtype shouldBe DType.Float32
        val boolTensor = Tensor(Shape1(Axis[A] -> 10)).fill(true)
        boolTensor.dtype shouldBe DType.Bool

      it("Fill tensors with widened types"):
        // Test byte defaults to int8
        val intTensorFromByte = Tensor(Shape2(Axis[A] -> 4, Axis[B] -> 5)).fill(42.toByte)
        intTensorFromByte.dtype shouldBe DType.Int8
        // Test double defaults to float64
        withJaxX64Support: // Enable float64 support in JAX
          val floatTensorFromDouble = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(3.14)
          floatTensorFromDouble.dtype shouldBe DType.Float64
    describe("Tensor fromArray"):
      it("fromArray with tensor types"):
        val intTensor = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(1, 2, 3))
        intTensor.dtype shouldBe DType.Int32
        val floatTensor = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 2)).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f))
        floatTensor.dtype shouldBe DType.Float32
      it("fromArray with widened types"):
        // Test short defaults to int8
        val intTensorFromShort = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(1.toByte, 2.toByte, 3.toByte))
        intTensorFromShort.dtype shouldBe DType.Int8
        // Test double defaults to float64
        withJaxX64Support: // Enable float64 support in JAX
          val floatTensorFromDouble = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 2)).fromArray(Array(1.0, 2.0, 3.0, 4.0))
          floatTensorFromDouble.dtype shouldBe DType.Float64

  describe("Overwrite default setings"):
    it("Change float default dtype from Float32 to Float64"):
      // Check fill
      withJaxX64Support: // Enable float64 support in JAX
        val t64 = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4), VType[Float64]).fill(3.14f)
        t64.dtype shouldBe DType.Float64
      // Check fromArray
      withJaxX64Support: // Enable float64 support in JAX
        val t64 = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 2), VType[Float64]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f))
        t64.dtype shouldBe DType.Float64

    it("Change double default dtype from Float64 to Float32"):
      // Check fill
      val floatTensorFromDouble = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4), VType[Float32]).fill(3.14)
      floatTensorFromDouble.dtype shouldBe DType.Float32
      // Check fromArray
      withJaxX64Support: // Enable float64 support in JAX
        val floatTensorFromDouble2 = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 2), VType[Float32]).fromArray(Array(1.0, 2.0, 3.0, 4.0))
        floatTensorFromDouble2.dtype shouldBe DType.Float32

  describe("fromFunction"):

    it("2D: identity matrix from indices"):
      val result = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 3)).fromFunction { idx =>
        if idx(Axis[A]) == idx(Axis[B]) then 1.0f else 0.0f
      }
      val expected = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 0.0f, 0.0f), Array(0.0f, 1.0f, 0.0f), Array(0.0f, 0.0f, 1.0f))
      )
      result shouldEqual expected

    it("2D: element values are row + col index"):
      val result = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fromFunction { idx =>
        (idx(Axis[A]) + idx(Axis[B])).toFloat
      }
      val expected = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(0.0f, 1.0f, 2.0f), Array(1.0f, 2.0f, 3.0f))
      )
      result shouldEqual expected

    it("1D: element values are their own index"):
      val result = Tensor(Shape1(Axis[A] -> 4)).fromFunction { idx =>
        idx(Axis[A]).toFloat
      }
      result shouldEqual Tensor1(Axis[A]).fromArray(Array(0.0f, 1.0f, 2.0f, 3.0f))

  describe("eye"):

    it("square: from two extents or from a shape"):
      val expected = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 0.0f), Array(0.0f, 1.0f))
      )
      Tensor2(Axis[A] -> 2, Axis[B] -> 2).eye shouldEqual expected
      Tensor2(Shape2(Axis[A] -> 2, Axis[B] -> 2)).eye shouldEqual expected

    it("square: from a single extent, the second axis is the primed copy of the first"):
      val result = Tensor2(Axis[A] -> 3).eye
      result.shape shouldEqual Shape2(Axis[A] -> 3, Axis[Prime[A]] -> 3)
      result shouldEqual Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 0.0f, 0.0f), Array(0.0f, 1.0f, 0.0f), Array(0.0f, 0.0f, 1.0f))
      )

    it("wide: more columns than rows, zero padded"):
      val expected = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 0.0f, 0.0f), Array(0.0f, 1.0f, 0.0f))
      )
      val result = Tensor2(Axis[A] -> 2, Axis[B] -> 3).eye
      result.shape shouldEqual Shape2(Axis[A] -> 2, Axis[B] -> 3)
      result shouldEqual expected

    it("tall: more rows than columns, truncating"):
      val expected = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 0.0f), Array(0.0f, 1.0f), Array(0.0f, 0.0f))
      )
      val result = Tensor2(Axis[A] -> 3, Axis[B] -> 2).eye
      result.shape shouldEqual Shape2(Axis[A] -> 3, Axis[B] -> 2)
      result shouldEqual expected

    it("defaults to Float32 and takes the vtype as an argument"):
      Tensor2(Axis[A] -> 2, Axis[B] -> 3).eye.dtype shouldBe DType.Float32
      Tensor2(Axis[A] -> 2, Axis[B] -> 3).eye(VType[Int32]).dtype shouldBe DType.Int32
      Tensor2(Shape2(Axis[A] -> 2, Axis[B] -> 3)).eye(VType[Int16]).dtype shouldBe DType.Int16
