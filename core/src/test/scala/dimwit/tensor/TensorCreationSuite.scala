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

  describe("fromArray with integers"):
    it("Short and Long arrays default to Int16 and Int64"):
      val shorts = Tensor1(Axis[A]).fromArray(Array[Short](-3, 0, 32767))
      shorts.dtype shouldBe DType.Int16
      shorts.toArray shouldBe Array[Short](-3, 0, 32767)
      withJaxX64Support:
        val longs = Tensor1(Axis[A]).fromArray(Array(-1L, Long.MaxValue))
        longs.dtype shouldBe DType.Int64
        longs.toArray shouldBe Array(-1L, Long.MaxValue)

    it("converts between integer widths"):
      Tensor1(Axis[A], VType[Int8]).fromArray(Array(-128, 0, 127)).toArray shouldBe Array[Byte](-128, 0, 127)
      Tensor1(Axis[A], VType[Int16]).fromArray(Array[Byte](1, -2)).toArray shouldBe Array[Short](1, -2)
      Tensor1(Axis[A], VType[Int32]).fromArray(Array(-5L, 7L)).toArray shouldBe Array(-5, 7)

    it("rejects values out of range of the value type"):
      val error = intercept[IllegalArgumentException]:
        Tensor1(Axis[A], VType[Int8]).fromArray(Array(1, 200))
      error.getMessage should include("Value 200 at index 1 is out of range for DType int8")

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

  describe("fromRange"):

    it("until: half-open interval"):
      val result = Tensor1(Axis[A]).fromRange(0 until 4)
      result.shape shouldEqual Shape1(Axis[A] -> 4)
      result shouldEqual Tensor1(Axis[A]).fromArray(Array(0, 1, 2, 3))
      Tensor1(Axis[A]).fromRange(2 until 5) shouldEqual Tensor1(Axis[A]).fromArray(Array(2, 3, 4))

    it("to: inclusive interval"):
      Tensor1(Axis[A]).fromRange(2 to 5) shouldEqual Tensor1(Axis[A]).fromArray(Array(2, 3, 4, 5))
      Tensor1(Axis[A]).fromRange(0 to 7 by 3) shouldEqual Tensor1(Axis[A]).fromArray(Array(0, 3, 6))

    it("by: stepped and negative steps count down"):
      Tensor1(Axis[A]).fromRange(0 until 7 by 3) shouldEqual Tensor1(Axis[A]).fromArray(Array(0, 3, 6))
      Tensor1(Axis[A]).fromRange(3 until 0 by -1) shouldEqual Tensor1(Axis[A]).fromArray(Array(3, 2, 1))
      Tensor1(Axis[A]).fromRange(10 to 0 by -3) shouldEqual Tensor1(Axis[A]).fromArray(Array(10, 7, 4, 1))

    it("empty range gives an empty vector"):
      Tensor1(Axis[A]).fromRange(0 until 0).shape shouldEqual Shape1(Axis[A] -> 0)
      Tensor1(Axis[A]).fromRange(5 until 2).shape shouldEqual Shape1(Axis[A] -> 0)

    it("defaults to Int32 and takes the vtype as an argument"):
      Tensor1(Axis[A]).fromRange(0 until 3).dtype shouldBe DType.Int32
      Tensor1(Axis[A]).fromRange(0 until 3, VType[Int16]).dtype shouldBe DType.Int16
      Tensor1(Axis[A]).fromRange(0 until 3, VType[Int16]).asInt32 shouldEqual Tensor1(Axis[A]).fromArray(Array(0, 1, 2))

    it("typed factory uses its vtype"):
      Tensor1(Axis[A], VType[Int16]).fromRange(0 until 3).dtype shouldBe DType.Int16
      Tensor1(Axis[A], VType[Int16]).fromRange(1 until 3).asInt32 shouldEqual Tensor1(Axis[A]).fromArray(Array(1, 2))

    it("rejects non-integer vtypes at compile time"):
      typeCheckErrors("Tensor1(Axis[A]).fromRange(0 until 3, VType[Float32])") should not be empty
      typeCheckErrors("Tensor1(Axis[A], VType[Float32]).fromRange(0 until 3)") should not be empty

    it("can be consumed as gather indices by select"):
      val t = Tensor1(Axis[A]).fromArray(Array(10.0f, 20.0f, 30.0f))
      t.select(Axis[A].at(Tensor1(Axis[B]).fromRange(0 until 3))) shouldEqual Tensor1(Axis[B]).fromArray(Array(10.0f, 20.0f, 30.0f))

  describe("linspace"):

    it("num evenly spaced values including the endpoint"):
      val result = Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 5)
      result.shape shouldEqual Shape1(Axis[A] -> 5)
      result shouldEqual Tensor1(Axis[A]).fromArray(Array(0.0f, 0.25f, 0.5f, 0.75f, 1.0f))
      Tensor1(Axis[A]).linspace(Tensor0(2.0f), Tensor0(3.0f), 3) shouldEqual Tensor1(Axis[A]).fromArray(Array(2.0f, 2.5f, 3.0f))

    it("endpoint = false excludes stop"):
      Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 4, endpoint = false) shouldEqual Tensor1(Axis[A]).fromArray(Array(0.0f, 0.25f, 0.5f, 0.75f))

    it("descending when start > stop"):
      Tensor1(Axis[A]).linspace(Tensor0(1.0f), Tensor0(0.0f), 3) shouldEqual Tensor1(Axis[A]).fromArray(Array(1.0f, 0.5f, 0.0f))

    it("num = 1 gives start, num = 0 gives an empty vector"):
      Tensor1(Axis[A]).linspace(Tensor0(3.0f), Tensor0(7.0f), 1) shouldEqual Tensor1(Axis[A]).fromArray(Array(3.0f))
      Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 0).shape shouldEqual Shape1(Axis[A] -> 0)

    it("value type is that of start and stop"):
      Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 3).dtype shouldBe DType.Float32
      Tensor1(Axis[A]).linspace(Tensor0(VType[Float16])(0.0f), Tensor0(VType[Float16])(1.0f), 3).dtype shouldBe DType.Float16
      withJaxX64Support:
        Tensor1(Axis[A]).linspace(Tensor0(0.0), Tensor0(1.0), 3).dtype shouldBe DType.Float64

    it("typed factory fixes the value type and accepts converted literals"):
      Tensor1(Axis[A], VType[Float16]).linspace(0.0f, 1.0f, 3).dtype shouldBe DType.Float16
      Tensor1(Axis[A], VType[Float32]).linspace(0.0f, 1.0f, 3, endpoint = false) shouldEqual Tensor1(Axis[A]).fromArray(Array(0.0f, 1.0f / 3.0f, 2.0f / 3.0f))

    it("start and stop can be data dependent"):
      val x = Tensor1(Axis[B]).fromArray(Array(4.0f, 2.0f, 8.0f))
      Tensor1(Axis[A]).linspace(x.min, x.max, 4) shouldEqual Tensor1(Axis[A]).fromArray(Array(2.0f, 4.0f, 6.0f, 8.0f))

    it("rejects non-floating and mixed value types at compile time"):
      typeCheckErrors("Tensor1(Axis[A]).linspace(Tensor0(0), Tensor0(3), 3)") should not be empty
      typeCheckErrors("Tensor1(Axis[A], VType[Int32]).linspace(Tensor0(0), Tensor0(3), 3)") should not be empty
      typeCheckErrors("Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0), 3)") should not be empty
