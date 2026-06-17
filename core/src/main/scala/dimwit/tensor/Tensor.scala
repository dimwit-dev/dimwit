package dimwit.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import dimwit.jax.Jax
import dimwit.jax.JaxDType
import dimwit.jax.Jax.PyDynamic
import dimwit.tensor.TypedIndex
import dimwit.tensor.{Label, Labels, VType}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random
import dimwit.stats.{Normal, Uniform}
import me.shadaj.scalapy.readwrite.Writer
import scala.reflect.ClassTag
import scala.annotation.unchecked.uncheckedVariance
import dimwit.Prime
import ShapeTypeHelpers.AxisIndex
import dimwit.hardware.Device
import me.shadaj.scalapy.readwrite.Writer.stringWriter.given
import dimwit.tensor.TensorOps.{IsBoolean, IsInteger, IsFloating}
import DType.*

class Tensor[T <: Tuple: Labels, V] private[dimwit] (
    private[dimwit] val jaxValue: Jax.PyDynamic
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromSeq[T](jaxValue.shape.as[Seq[Int]])
  lazy val vtype: VType[V] = VType(this)

  lazy val device: Device = Device(jaxValue.device)

  def asType[V2](vtype: VType[V2]): Tensor[T, V2] = new Tensor(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(vtype.dtype)))

  def toDevice(newDevice: Device): Tensor[T, V] = new Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.toJaxDevice))

  override def equals(other: Any): Boolean =
    other match
      case that: Tensor[?, ?] => Jax.jnp.array_equal(this.jaxValue, that.jaxValue).item().as[Boolean]
      case _                  => false

  override def hashCode(): Int = jaxValue.block_until_ready().tobytes().hashCode()

  override def toString: String =
    jaxTypeName match
      case Jax.ArrayTypeName =>
        jaxValue.block_until_ready().toString()
      case Jax.BatchTracerName =>
        s"TracerTensor(${shape.toString})"
      case _ => jaxValue.toString()

  def extent[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): AxisExtent[L] =
    shape.extent(axis)

  private val jaxTypeName: String = py.Dynamic.global.`type`(jaxValue).`__name__`.as[String]
  lazy val isTracer: Boolean =
    val jaxCoreTracer = py.module("jax.core").Tracer
    py.Dynamic.global.isinstance(jaxValue, jaxCoreTracer).as[Boolean]

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  case class DefaultsFactory[T <: Tuple: Labels](shape: Shape[T]):

    def fill(value: Float): Tensor[T, Float32] = Tensor(shape, VType[Float32]).fill(value)
    def fill(value: Double): Tensor[T, Float64] = Tensor(shape, VType[Float64]).fill(value)
    def fromArray(values: Array[Float]): Tensor[T, Float32] = Tensor(shape, VType[Float32]).fromArray(values)
    def fromArray(values: Array[Double]): Tensor[T, Float64] = Tensor(shape, VType[Float64]).fromArray(values)

    def fill(value: Byte): Tensor[T, Int8] = Tensor(shape, VType[Int8]).fill(value)
    def fill(value: Short): Tensor[T, Int16] = Tensor(shape, VType[Int16]).fill(value)
    def fill(value: Int): Tensor[T, Int32] = Tensor(shape, VType[Int32]).fill(value)
    def fill(value: Long): Tensor[T, Int64] = Tensor(shape, VType[Int64]).fill(value)
    def fromArray(values: Array[Byte]): Tensor[T, Int8] = Tensor(shape, VType[Int8]).fromArray(values)
    def fromArray(values: Array[Short]): Tensor[T, Int16] = Tensor(shape, VType[Int16]).fromArray(values)
    def fromArray(values: Array[Int]): Tensor[T, Int32] = Tensor(shape, VType[Int32]).fromArray(values)
    def fromArray(values: Array[Long]): Tensor[T, Int64] = Tensor(shape, VType[Int64]).fromArray(values)

    def fill(value: Boolean): Tensor[T, Bool] = Tensor(shape, VType[Bool]).fill(value)
    def fromArray(values: Array[Boolean]): Tensor[T, Bool] = Tensor(shape, VType[Bool]).fromArray(values)

    @targetName("fromFunctionFloat")
    def fromFunction(f: TypedIndex[T] => Float): Tensor[T, Float32] =
      Tensor(shape, VType[Float32]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionDouble")
    def fromFunction(f: TypedIndex[T] => Double): Tensor[T, Float64] =
      Tensor(shape, VType[Float64]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionByte")
    def fromFunction(f: TypedIndex[T] => Byte): Tensor[T, Int8] =
      Tensor(shape, VType[Int8]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionShort")
    def fromFunction(f: TypedIndex[T] => Short): Tensor[T, Int16] =
      Tensor(shape, VType[Int16]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionInt")
    def fromFunction(f: TypedIndex[T] => Int): Tensor[T, Int32] =
      Tensor(shape, VType[Int32]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionLong")
    def fromFunction(f: TypedIndex[T] => Long): Tensor[T, Int64] =
      Tensor(shape, VType[Int64]).fromArray(Tensor.tabulate(shape.dimensions, f))
    @targetName("fromFunctionBoolean")
    def fromFunction(f: TypedIndex[T] => Boolean): Tensor[T, Bool] =
      Tensor(shape, VType[Bool]).fromArray(Tensor.tabulate(shape.dimensions, f))

  case class TypedFactory[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]):

    // --- Boolean ---
    def fill(value: Boolean)(using IsBoolean[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

    // --- Integer ---
    def fill(value: Byte)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Short)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value.toInt, dtype = vtype.dtype.jaxType))
    def fill(value: Int)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Long)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

    // --- Floating ---
    def fill(value: Float)(using IsFloating[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Double)(using IsFloating[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

  case class LikeFactory[T <: Tuple: Labels, V](val other: Tensor[T, V]):

    def fill(value: Boolean): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)

    def fill(value: Byte): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Short): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value.toInt, dtype = other.dtype.jaxType))
    def fill(value: Int): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Long): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)

    def fill(value: Float): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Double): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)

  /** Computes per-element values via eager Scala-side iteration. Used by [[DefaultsFactory.fromFunction]]. */
  private[tensor] def tabulate[T <: Tuple, V: scala.reflect.ClassTag](dims: List[Int], f: TypedIndex[T] => V): Array[V] =
    val strides = dims.scanRight(1)(_ * _).tail
    Array.tabulate(dims.product) { flatIdx =>
      f(TypedIndex[T](IndexedSeq.tabulate(dims.length)(d => (flatIdx / strides(d)) % dims(d))))
    }

  private[dimwit] def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)

  def apply[T <: Tuple: Labels](shape: Shape[T]): DefaultsFactory[T] = DefaultsFactory(shape)
  def apply[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): TypedFactory[T, V] = TypedFactory(shape, vtype)
  def like[T <: Tuple: Labels, V](template: Tensor[T, V]): Tensor.LikeFactory[T, V] = Tensor.LikeFactory(template)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  given boolean2BooleanTensor[V: IsBoolean]: Conversion[Boolean, Tensor0[V]] with
    def apply(value: Boolean): Tensor0[V] = Tensor0(VType[V])(value)

  given byte2IntegerTensor[V: IsInteger]: Conversion[Byte, Tensor0[V]] with
    def apply(value: Byte): Tensor0[V] = Tensor0(VType[V])(value)

  given short2IntegerTensor[V: IsInteger]: Conversion[Short, Tensor0[V]] with
    def apply(value: Short): Tensor0[V] = Tensor0(VType[V])(value)

  given int2IntegerTensor[V: IsInteger]: Conversion[Int, Tensor0[V]] with
    def apply(value: Int): Tensor0[V] = Tensor0(VType[V])(value)

  given int2FloatingTensor[V: IsFloating]: Conversion[Int, Tensor0[V]] with
    def apply(value: Int): Tensor0[V] = Tensor0(VType[V])(value.toFloat)

  given long2IntegerTensor[V: IsInteger]: Conversion[Long, Tensor0[V]] with
    def apply(value: Long): Tensor0[V] = Tensor0(VType[V])(value)

  given float2FloatingTensor[V: IsFloating]: Conversion[Float, Tensor0[V]] with
    def apply(value: Float): Tensor0[V] = Tensor0(VType[V])(value)

  given double2FloatingTensor[V: IsFloating]: Conversion[Double, Tensor0[V]] with
    def apply(value: Double): Tensor0[V] = Tensor0(VType[V])(value)

  object DefaultsFactory:

    def apply(value: Boolean): Tensor0[Bool] = Tensor0(VType[Bool])(value)

    def apply(value: Byte): Tensor0[Int8] = Tensor0(VType[Int8])(value)
    def apply(value: Short): Tensor0[Int16] = Tensor0(VType[Int16])(value)
    def apply(value: Int): Tensor0[Int32] = Tensor0(VType[Int32])(value)
    def apply(value: Long): Tensor0[Int64] = Tensor0(VType[Int64])(value)

    def apply(value: Float): Tensor0[Float32] = Tensor0(VType[Float32])(value)
    def apply(value: Double): Tensor0[Float64] = Tensor0(VType[Float64])(value)

  case class TypedFactory[V](vtype: VType[V]):

    // --- Boolean ---
    def apply(value: Boolean)(using IsBoolean[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))

    // --- Integer ---
    def apply(value: Byte)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Short)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value.toInt, dtype = vtype.dtype.jaxType))
    def apply(value: Int)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Long)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))

    // --- Floating ---
    def apply(value: Float)(using IsFloating[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Double)(using IsFloating[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))

  export DefaultsFactory.*
  def apply[V](vtype: VType[V]): TypedFactory[V] = TypedFactory(vtype)

  def like[V: Writer](template: Tensor0[V])(value: V): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))
  def likeDType[V, T <: Tuple](template: Tensor[T, V])(value: Float): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))

  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)

object Tensor1:

  case class DefaultsFactory[L: Label](axis: Axis[L]):

    // --- Boolean ---
    def fromArray(values: Array[Boolean]): Tensor1[L, Bool] = Tensor1(axis, VType[Bool]).fromArray(values)

    // --- Integer ---
    def fromArray(values: Array[Byte]): Tensor1[L, Int8] = Tensor1(axis, VType[Int8]).fromArray(values)
    def fromArray(values: Array[Short]): Tensor1[L, Int16] = Tensor1(axis, VType[Int16]).fromArray(values)
    def fromArray(values: Array[Int]): Tensor1[L, Int32] = Tensor1(axis, VType[Int32]).fromArray(values)
    def fromArray(values: Array[Long]): Tensor1[L, Int64] = Tensor1(axis, VType[Int64]).fromArray(values)

    // --- Floating ---
    def fromArray(values: Array[Float]): Tensor1[L, Float32] = Tensor1(axis, VType[Float32]).fromArray(values)
    def fromArray(values: Array[Double]): Tensor1[L, Float64] = Tensor1(axis, VType[Float64]).fromArray(values)

  case class TypedFactory[L: Label, V](axis: Axis[L], vtype: VType[V]):

    // --- Boolean ---
    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)

    // --- Integer ---
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)

    // --- Floating ---
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)

  def apply[L: Label](axis: Axis[L]): DefaultsFactory[L] = DefaultsFactory(axis)
  def apply[L: Label, V](axis: Axis[L], vtype: VType[V]): TypedFactory[L, V] = TypedFactory(axis, vtype)

object Tensor2:

  type Array2D[V] = Array[Array[V]]

  case class DefaultsFactory[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]):

    def fromArray(values: Array2D[Boolean]): Tensor2[L1, L2, Bool] = Tensor2(axis1, axis2, VType[Bool]).fromArray(values)

    def fromArray(values: Array2D[Byte]): Tensor2[L1, L2, Int8] = Tensor2(axis1, axis2, VType[Int8]).fromArray(values)
    def fromArray(values: Array2D[Short]): Tensor2[L1, L2, Int16] = Tensor2(axis1, axis2, VType[Int16]).fromArray(values)
    def fromArray(values: Array2D[Int]): Tensor2[L1, L2, Int32] = Tensor2(axis1, axis2, VType[Int32]).fromArray(values)
    def fromArray(values: Array2D[Long]): Tensor2[L1, L2, Int64] = Tensor2(axis1, axis2, VType[Int64]).fromArray(values)

    def fromArray(values: Array2D[Float]): Tensor2[L1, L2, Float32] = Tensor2(axis1, axis2, VType[Float32]).fromArray(values)
    def fromArray(values: Array2D[Double]): Tensor2[L1, L2, Float64] = Tensor2(axis1, axis2, VType[Float64]).fromArray(values)

  case class TypedFactory[L1: Label, L2: Label, V](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]):

    private def createShape[V](values: Array2D[V]): Shape2[L1, L2] = Shape2(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length))

    // --- Boolean ---
    def fromArray(values: Array2D[Boolean])(using IsBoolean[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)

    // --- Integer ---
    def fromArray(values: Array2D[Byte])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Short])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Int])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Long])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)

    // --- Floating ---
    def fromArray(values: Array2D[Float])(using IsFloating[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Double])(using IsFloating[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)

  def apply[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): DefaultsFactory[L1, L2] = DefaultsFactory(axis1, axis2)
  def apply[L1: Label, L2: Label, V](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): TypedFactory[L1, L2, V] = TypedFactory(axis1, axis2, vtype)

  private def eyeImpl[L: Label, V](dim: AxisExtent[L], vtype: VType[V]): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.eye(dim.size, dtype = vtype.dtype.jaxType))
  def eye[L: Label](dim: AxisExtent[L]): Tensor2[L, Prime[L], Float32] = eyeImpl(dim, VType[Float32])
  def eye[L: Label, V](dim: AxisExtent[L], vtype: VType[V]): Tensor2[L, Prime[L], V] = eyeImpl(dim, vtype)
  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  type Array3D[V] = Array[Array[Array[V]]]

  case class DefaultsFactory[L1: Label, L2: Label, L3: Label](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3]):

    def fromArray(values: Array3D[Boolean]): Tensor3[L1, L2, L3, Bool] = Tensor3(axis1, axis2, axis3, VType[Bool]).fromArray(values)

    def fromArray(values: Array3D[Byte]): Tensor3[L1, L2, L3, Int8] = Tensor3(axis1, axis2, axis3, VType[Int8]).fromArray(values)
    def fromArray(values: Array3D[Short]): Tensor3[L1, L2, L3, Int16] = Tensor3(axis1, axis2, axis3, VType[Int16]).fromArray(values)
    def fromArray(values: Array3D[Int]): Tensor3[L1, L2, L3, Int32] = Tensor3(axis1, axis2, axis3, VType[Int32]).fromArray(values)
    def fromArray(values: Array3D[Long]): Tensor3[L1, L2, L3, Int64] = Tensor3(axis1, axis2, axis3, VType[Int64]).fromArray(values)

    def fromArray(values: Array3D[Float]): Tensor3[L1, L2, L3, Float32] = Tensor3(axis1, axis2, axis3, VType[Float32]).fromArray(values)
    def fromArray(values: Array3D[Double]): Tensor3[L1, L2, L3, Float64] = Tensor3(axis1, axis2, axis3, VType[Float64]).fromArray(values)

  case class TypedFactory[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[V]):

    private def createShape[V](values: Array3D[V]): Shape3[L1, L2, L3] = Shape3(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length), AxisExtent(axis3, values.head.head.length))

    // --- Boolean ---
    def fromArray(values: Array3D[Boolean])(using IsBoolean[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)

    // --- Integer ---
    def fromArray(values: Array3D[Byte])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Short])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Int])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Long])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)

    // --- Floating ---
    def fromArray(values: Array3D[Float])(using IsFloating[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Double])(using IsFloating[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)

  def apply[L1: Label, L2: Label, L3: Label](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3]): DefaultsFactory[L1, L2, L3] = DefaultsFactory(axis1, axis2, axis3)

  def apply[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[V]): TypedFactory[L1, L2, L3, V] = TypedFactory(axis1, axis2, axis3, vtype)
