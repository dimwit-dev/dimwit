package dimwit.tensor

import dimwit.*
import dimwit.DType.*
import dimwit.DType.given

trait A derives Label
trait B derives Label

def main =
  {
    val t0_i32_1: Tensor0[Int32] = Tensor0(0)
    val t0_f32_1: Tensor0[Float32] = Tensor0(0f)
    val t0_f32_2: Tensor0[Float32] = Tensor0(VType[Float32])(0.0)
    val t0_f64_1: Tensor0[Float64] = Tensor0(0.0)
  }
  {
    val array = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val t_i32: Tensor1[A, Int32] = Tensor1(Axis[A]).fromArray(array.map(_.toInt))
    val t_i64: Tensor1[A, Int64] = Tensor1(Axis[A], VType[Int64]).fromArray(array.map(_.toInt))
    val t_f32: Tensor1[A, Float32] = Tensor1(Axis[A]).fromArray(array)
    val t_f64: Tensor1[A, Float64] = Tensor1(Axis[A]).fromArray(array.map(_.toDouble))
    val i64Factory = Tensor1(Axis[A], VType[Int64])
    i64Factory.fromArray(array.map(_.toInt))
  }
  {
    val array = Array(
      Array(1.0f, 2.0f, 3.0f),
      Array(4.0f, 5.0f, 6.0f)
    )
    val t_f32: Tensor2[A, B, Float32] = Tensor2(Axis[A], Axis[B]).fromArray(array)
    val t_f64: Tensor2[A, B, Float64] = Tensor2(Axis[A], Axis[B], VType[Float64]).fromArray(array)
  }
  {
    val shape = Shape2(Axis[A] -> 2, Axis[B] -> 3)
    val t_f32_1: Tensor[(A, B), Float32] = Tensor(shape).fill(0f)
    val t_f32_2: Tensor[(A, B), Float32] = Tensor(shape, VType[Float32]).fill(0.0)
    val t_f64_1: Tensor[(A, B), Float64] = Tensor(shape).fill(0.0)
    val t_f64_2: Tensor[(A, B), Float64] = Tensor(shape, VType[Float64]).fill(0f)
  }
  {
    val shape = Shape2(Axis[A] -> 2, Axis[B] -> 3)
    val array = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val t_f32_1: Tensor[(A, B), Float32] = Tensor(shape).fromArray(array)
    val t_f32_2: Tensor[(A, B), Float32] = Tensor(shape, VType[Float32]).fromArray(array)
    val t_f64_1: Tensor[(A, B), Float64] = Tensor(shape).fromArray(array.map(_.toDouble))
    val t_f64_2: Tensor[(A, B), Float64] = Tensor(shape, VType[Float64]).fromArray(array.map(_.toDouble))
  }
  {
    val shape = Shape2(Axis[A] -> 2, Axis[B] -> 3)
    val array = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val template = Tensor(shape).fromArray(array)
    val t_f32_1: Tensor[(A, B), Float32] = Tensor.like(template).fromArray(array)
    val t_f64_1: Tensor[(A, B), Float32] = Tensor.like(template).fromArray(array.map(_.toDouble))
  }
  {
    val shape = Shape2(Axis[A] -> 2, Axis[B] -> 3)
    val array = Array(1, 2, 3, 4, 5, 6)
    val template = Tensor(shape).fromArray(array)
    val t_1: Tensor[(A, B), Int32] = Tensor.like(template).fromArray(array)
    val t_2: Tensor[(A, B), Int32] = Tensor.like(template).fromArray(array.map(_.toLong))
  }
  {
    val array = Array(1, 2, 3, 4, 5, 6)
    val t_i32: Tensor1[A, Int32] = Tensor1(Axis[A]).fromArray(array)
    val t_b: Tensor[Tuple1[A], Bool] = t_i32.asBool
    val t_f32_1: Tensor[Tuple1[A], Float32] = t_i32.asFloat32
    val t_f32_2: Tensor[Tuple1[A], Float32] = t_i32.asFloat[Float32]
    val t_f64: Tensor[Tuple1[A], Float64] = t_i32.asFloat[Float64]

    class ExampleScope[V: IsFloating]:
      def castToFloat(t: Tensor0[Int32]): Tensor0[V] = t.asFloat[V]
  }
