package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class TensorCreationSuite extends AnyFunSpec with Matchers:

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
    it("Change double default dtype from Float64 to Float32"):
      given ExecutionType[Double] = ExecutionTypeFor[Double](DType.Float32)
      // Check fill
      val floatTensorFromDouble = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(3.14)
      floatTensorFromDouble.dtype shouldBe DType.Float32
      // Check fromArray
      withJaxX64Support: // Enable float64 support in JAX
        val floatTensorFromDouble2 = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 2)).fromArray(Array(1.0, 2.0, 3.0, 4.0))
        floatTensorFromDouble2.dtype shouldBe DType.Float32
