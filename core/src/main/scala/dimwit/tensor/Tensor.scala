package dimwit.tensor

import dimwit.prime.Prime
import dimwit.hardware.Device
import dimwit.jax.Jax
import dimwit.jax.Jax.PyDynamic
import dimwit.jax.JaxDType
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.TensorOps.IsBoolean
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TypedIndex
import dimwit.tensor.VType
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.targetName
import scala.reflect.ClassTag

import ShapeTypeHelpers.AxisIndex
import DType.*

/** A tensor with a fixed shape and data type.
  *
  * @param T The shape of the tensor, represented as a tuple of axis labels.
  * @param V The data type of the tensor elements.
  */
class Tensor[T <: Tuple: Labels, V] private[dimwit] (
    private[dimwit] val jaxValue: Jax.PyDynamic
):

  /** The labels of the tensor's axes. */
  lazy val axes: List[String] = shape.labels

  /** The shape of the tensor. */
  lazy val shape: Shape[T] = Shape.fromSeq[T](jaxValue.shape.as[Seq[Int]])

  /** The data type of the underlying Jax tensor. */
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)

  /** The value type of the tensor (static type information) */
  lazy val vtype: VType[V] = VType(this)

  /** The device on which the tensor is stored. */
  lazy val device: Device = Device(jaxValue.device)

  /** Converts the tensor to the given vtype.
    */
  def asType[V2](vtype: VType[V2]): Tensor[T, V2] = new Tensor(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(vtype.dtype)))

  /** Moves the tensor to a different device. */
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

  /** Returns the @AxisExtent of the specified axis in the tensor's shape. */
  def extent[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): AxisExtent[L] =
    shape.extent(axis)

  private val jaxTypeName: String = py.Dynamic.global.`type`(jaxValue).`__name__`.as[String]
  lazy val isTracer: Boolean =
    val jaxCoreTracer = py.module("jax.core").Tracer
    py.Dynamic.global.isinstance(jaxValue, jaxCoreTracer).as[Boolean]

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  /** Factory for createing tensors with a specific shape.
    *
    * @param shape The shape of the tensor to create.
    */
  class ShapedFactory[T <: Tuple: Labels](shape: Shape[T]):

    /** Ceates a tensor filled with the specified value. */
    def fill(value: Float): Tensor[T, Float32] = Tensor(shape, VType[Float32]).fill(value)
    def fill(value: Double): Tensor[T, Float64] = Tensor(shape, VType[Float64]).fill(value)
    def fill(value: Byte): Tensor[T, Int8] = Tensor(shape, VType[Int8]).fill(value)
    def fill(value: Short): Tensor[T, Int16] = Tensor(shape, VType[Int16]).fill(value)
    def fill(value: Int): Tensor[T, Int32] = Tensor(shape, VType[Int32]).fill(value)
    def fill(value: Long): Tensor[T, Int64] = Tensor(shape, VType[Int64]).fill(value)
    def fill(value: Boolean): Tensor[T, Bool] = Tensor(shape, VType[Bool]).fill(value)

    /** Creates a tensor from an array of values.
      * The array must have the same number of elements
      * as the product of the dimensions in the shape.
      */
    def fromArray(values: Array[Float]): Tensor[T, Float32] = Tensor(shape, VType[Float32]).fromArray(values)
    def fromArray(values: Array[Double]): Tensor[T, Float64] = Tensor(shape, VType[Float64]).fromArray(values)
    def fromArray(values: Array[Byte]): Tensor[T, Int8] = Tensor(shape, VType[Int8]).fromArray(values)
    def fromArray(values: Array[Short]): Tensor[T, Int16] = Tensor(shape, VType[Int16]).fromArray(values)
    def fromArray(values: Array[Int]): Tensor[T, Int32] = Tensor(shape, VType[Int32]).fromArray(values)
    def fromArray(values: Array[Long]): Tensor[T, Int64] = Tensor(shape, VType[Int64]).fromArray(values)
    def fromArray(values: Array[Boolean]): Tensor[T, Bool] = Tensor(shape, VType[Bool]).fromArray(values)

    /** Creates a tensor by computing each element
      * using the provided function.
      */
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

  /** Factory for creating tensors with a specific shape and a given value type
    *
    * @param shape The shape of the tensor to create
    * @param vtype The value type of the tensor to create
    */
  class ShapedTypedFactory[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]):

    /** @see [[ShapedFactory.fill]] */
    def fill(value: Boolean)(using IsBoolean[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Byte)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Short)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value.toInt, dtype = vtype.dtype.jaxType))
    def fill(value: Int)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Long)(using IsInteger[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Float)(using IsFloating[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
    def fill(value: Double)(using IsFloating[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))

    /** @see [[ShapedFactory.fromArray]] */
    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

  /** Factory for creating tensors with the same shape and value type as another tensor.
    *
    * @param other The tensor to use as a template for the new tensor.
    */
  class LikeFactory[T <: Tuple: Labels, V](val other: Tensor[T, V]):

    /** @see [[ShapedFactory.fill]] */
    def fill(value: Boolean): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Byte): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Short): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value.toInt, dtype = other.dtype.jaxType))
    def fill(value: Int): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Long): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Float): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))
    def fill(value: Double): Tensor[T, V] = Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))

    /** @see [[ShapedFactory.fromArray]] */
    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor[T, V] = ArrayWriter.fromArray[T, V](other.shape, values)

  /** Computes per-element values via eager Scala-side iteration. Used by [[ShapedFactory.fromFunction]]. */
  private[tensor] def tabulate[T <: Tuple, V: scala.reflect.ClassTag](dims: List[Int], f: TypedIndex[T] => V): Array[V] =
    val strides = dims.scanRight(1)(_ * _).tail
    Array.tabulate(dims.product) { flatIdx =>
      f(TypedIndex[T](IndexedSeq.tabulate(dims.length)(d => (flatIdx / strides(d)) % dims(d))))
    }

  private[dimwit] def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)

  /** Use the [[ShapedFactory]] to create a tensor */
  def apply[T <: Tuple: Labels](shape: Shape[T]): ShapedFactory[T] = ShapedFactory(shape)

  /** Use the [[ShapedTypedFactory]] to create a tensor */
  def apply[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): ShapedTypedFactory[T, V] = ShapedTypedFactory(shape, vtype)

  /** Use the [[LikeFactory]] to create a tensor */
  def like[T <: Tuple: Labels, V](template: Tensor[T, V]): LikeFactory[T, V] = LikeFactory(template)

/** Type aliases for tensors of different ranks. */
type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

/** Companion object for Tensors of rank 0 (scalars).
  *  Provides factory methods for creating tensors of rank 0 with various value types.
  */
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

  object Value0Factory:

    def apply(value: Boolean): Tensor0[Bool] = Tensor0(VType[Bool])(value)
    def apply(value: Byte): Tensor0[Int8] = Tensor0(VType[Int8])(value)
    def apply(value: Short): Tensor0[Int16] = Tensor0(VType[Int16])(value)
    def apply(value: Int): Tensor0[Int32] = Tensor0(VType[Int32])(value)
    def apply(value: Long): Tensor0[Int64] = Tensor0(VType[Int64])(value)
    def apply(value: Float): Tensor0[Float32] = Tensor0(VType[Float32])(value)
    def apply(value: Double): Tensor0[Float64] = Tensor0(VType[Float64])(value)

  class Value0TypedFactory[V](vtype: VType[V]):

    def apply(value: Boolean)(using IsBoolean[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Byte)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Short)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value.toInt, dtype = vtype.dtype.jaxType))
    def apply(value: Int)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Long)(using IsInteger[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Float)(using IsFloating[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))
    def apply(value: Double)(using IsFloating[V]): Tensor0[V] = Tensor(Jax.jnp.array(value, dtype = vtype.dtype.jaxType))

  export Value0Factory.*
  def apply[V](vtype: VType[V]): Value0TypedFactory[V] = Value0TypedFactory(vtype)

  def like[V: Writer](template: Tensor0[V])(value: V): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))
  def likeDType[V, T <: Tuple](template: Tensor[T, V])(value: Float): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))

  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)

/** Companion object for Tensors of rank 1 (vectors).
  *  Provides factory methods for creating tensors of rank 1 with various value types.
  */
object Tensor1:

  class AxisFactory[L: Label](axis: Axis[L]):

    def fromArray(values: Array[Boolean]): Tensor1[L, Bool] = Tensor1(axis, VType[Bool]).fromArray(values)
    def fromArray(values: Array[Byte]): Tensor1[L, Int8] = Tensor1(axis, VType[Int8]).fromArray(values)
    def fromArray(values: Array[Short]): Tensor1[L, Int16] = Tensor1(axis, VType[Int16]).fromArray(values)
    def fromArray(values: Array[Int]): Tensor1[L, Int32] = Tensor1(axis, VType[Int32]).fromArray(values)
    def fromArray(values: Array[Long]): Tensor1[L, Int64] = Tensor1(axis, VType[Int64]).fromArray(values)
    def fromArray(values: Array[Float]): Tensor1[L, Float32] = Tensor1(axis, VType[Float32]).fromArray(values)
    def fromArray(values: Array[Double]): Tensor1[L, Float64] = Tensor1(axis, VType[Float64]).fromArray(values)

  class AxisTypedFactory[L: Label, V](axis: Axis[L], vtype: VType[V]):

    def fromArray(values: Array[Boolean])(using IsBoolean[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Byte])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Short])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Int])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Long])(using IsInteger[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Float])(using IsFloating[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)
    def fromArray(values: Array[Double])(using IsFloating[V]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(axis -> values.length), values)

  def apply[L: Label](axis: Axis[L]): AxisFactory[L] = AxisFactory(axis)
  def apply[L: Label, V](axis: Axis[L], vtype: VType[V]): AxisTypedFactory[L, V] = AxisTypedFactory(axis, vtype)

  def apply[L: Label](axisExtent: AxisExtent[L]): Tensor.ShapedFactory[Tuple1[L]] = Tensor.ShapedFactory(Shape(axisExtent))
  def apply[L: Label, V](axisExtent: AxisExtent[L], vtype: VType[V]): Tensor.ShapedTypedFactory[Tuple1[L], V] = Tensor.ShapedTypedFactory(Shape(axisExtent), vtype)

/* Companion object for Tensors of rank 2 (matrices).
 *  Provides factory methods for creating tensors of rank 2 with various value types.
 */
object Tensor2:

  type Array2D[V] = Array[Array[V]]

  class Axes2Factory[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]):

    def fromArray(values: Array2D[Boolean]): Tensor2[L1, L2, Bool] = Tensor2(axis1, axis2, VType[Bool]).fromArray(values)
    def fromArray(values: Array2D[Byte]): Tensor2[L1, L2, Int8] = Tensor2(axis1, axis2, VType[Int8]).fromArray(values)
    def fromArray(values: Array2D[Short]): Tensor2[L1, L2, Int16] = Tensor2(axis1, axis2, VType[Int16]).fromArray(values)
    def fromArray(values: Array2D[Int]): Tensor2[L1, L2, Int32] = Tensor2(axis1, axis2, VType[Int32]).fromArray(values)
    def fromArray(values: Array2D[Long]): Tensor2[L1, L2, Int64] = Tensor2(axis1, axis2, VType[Int64]).fromArray(values)
    def fromArray(values: Array2D[Float]): Tensor2[L1, L2, Float32] = Tensor2(axis1, axis2, VType[Float32]).fromArray(values)
    def fromArray(values: Array2D[Double]): Tensor2[L1, L2, Float64] = Tensor2(axis1, axis2, VType[Float64]).fromArray(values)

  class Axes2TypedFactory[L1: Label, L2: Label, V](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]):

    private def createShape[V](values: Array2D[V]): Shape2[L1, L2] = Shape2(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length))

    def fromArray(values: Array2D[Boolean])(using IsBoolean[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Byte])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Short])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Int])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Long])(using IsInteger[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Float])(using IsFloating[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)
    def fromArray(values: Array2D[Double])(using IsFloating[V]): Tensor2[L1, L2, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten)

  /** Factory for creating matrices of a known shape, i.e. the [[Tensor.ShapedFactory]] extended by
    * the creation methods that only make sense for a matrix.
    *
    * @param matrixShape The shape of the matrix to create.
    */
  class Shaped2Factory[L1: Label, L2: Label](shape: Shape2[L1, L2]) extends Tensor.ShapedFactory[(L1, L2)](shape):

    def eye: Tensor2[L1, L2, Float32] = eye(VType[Float32])
    def eye[V](vtype: VType[V]): Tensor2[L1, L2, V] =
      Tensor(Jax.jnp.eye(shape.dimensions(0), shape.dimensions(1), dtype = vtype.dtype.jaxType))

  def apply[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Axes2Factory[L1, L2] = Axes2Factory(axis1, axis2)
  def apply[L1: Label, L2: Label, V](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Axes2TypedFactory[L1, L2, V] = Axes2TypedFactory(axis1, axis2, vtype)

  /** Creates the factory for a square matrix, whose second axis is the primed copy of the given one,
    * e.g. `Tensor2(Axis[A] -> 3).eye` has type `Tensor2[A, Prime[A], Float32]`.
    */
  def apply[L: Label](axisExtent: AxisExtent[L]): Shaped2Factory[L, Prime[L]] = Tensor2(axisExtent, Axis[Prime[L]] -> axisExtent.size)

  def apply[L1: Label, L2: Label](shape: Shape2[L1, L2]): Shaped2Factory[L1, L2] = Shaped2Factory(shape)
  def apply[L1: Label, L2: Label](axisExtent1: AxisExtent[L1], axisExtent2: AxisExtent[L2]): Shaped2Factory[L1, L2] = Tensor2(Shape2(axisExtent1, axisExtent2))
  def apply[L1: Label, L2: Label, V](axisExtent1: AxisExtent[L1], axisExtent2: AxisExtent[L2], vtype: VType[V]): Tensor.ShapedTypedFactory[Tuple2[L1, L2], V] = Tensor.ShapedTypedFactory(Shape(axisExtent1, axisExtent2), vtype)

  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.diag(diag.jaxValue))

/** Companion object for Tensors of rank 3.
  *  Provides factory methods for creating tensors of rank 3 with various value types.
  */
object Tensor3:

  type Array3D[V] = Array[Array[Array[V]]]

  class Axes3Factory[L1: Label, L2: Label, L3: Label](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3]):

    def fromArray(values: Array3D[Boolean]): Tensor3[L1, L2, L3, Bool] = Tensor3(axis1, axis2, axis3, VType[Bool]).fromArray(values)
    def fromArray(values: Array3D[Byte]): Tensor3[L1, L2, L3, Int8] = Tensor3(axis1, axis2, axis3, VType[Int8]).fromArray(values)
    def fromArray(values: Array3D[Short]): Tensor3[L1, L2, L3, Int16] = Tensor3(axis1, axis2, axis3, VType[Int16]).fromArray(values)
    def fromArray(values: Array3D[Int]): Tensor3[L1, L2, L3, Int32] = Tensor3(axis1, axis2, axis3, VType[Int32]).fromArray(values)
    def fromArray(values: Array3D[Long]): Tensor3[L1, L2, L3, Int64] = Tensor3(axis1, axis2, axis3, VType[Int64]).fromArray(values)
    def fromArray(values: Array3D[Float]): Tensor3[L1, L2, L3, Float32] = Tensor3(axis1, axis2, axis3, VType[Float32]).fromArray(values)
    def fromArray(values: Array3D[Double]): Tensor3[L1, L2, L3, Float64] = Tensor3(axis1, axis2, axis3, VType[Float64]).fromArray(values)

  class Axes3TypedFactory[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[V]):

    private def createShape[V](values: Array3D[V]): Shape3[L1, L2, L3] = Shape3(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length), AxisExtent(axis3, values.head.head.length))
    def fromArray(values: Array3D[Boolean])(using IsBoolean[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Byte])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Short])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Int])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Long])(using IsInteger[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Float])(using IsFloating[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)
    def fromArray(values: Array3D[Double])(using IsFloating[V]): Tensor3[L1, L2, L3, V] = Tensor(createShape(values), VType[V]).fromArray(values.flatten.flatten)

  def apply[L1: Label, L2: Label, L3: Label](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3]): Axes3Factory[L1, L2, L3] = Axes3Factory(axis1, axis2, axis3)
  def apply[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[V]): Axes3TypedFactory[L1, L2, L3, V] = Axes3TypedFactory(axis1, axis2, axis3, vtype)

  def apply[L1: Label, L2: Label, L3: Label](axisExtent1: AxisExtent[L1], axisExtent2: AxisExtent[L2], axisExtent3: AxisExtent[L3]): Tensor.ShapedFactory[Tuple3[L1, L2, L3]] = Tensor.ShapedFactory(Shape(axisExtent1, axisExtent2, axisExtent3))
  def apply[L1: Label, L2: Label, L3: Label, V](axisExtent1: AxisExtent[L1], axisExtent2: AxisExtent[L2], axisExtent3: AxisExtent[L3], vtype: VType[V]): Tensor.ShapedTypedFactory[Tuple3[L1, L2, L3], V] = Tensor.ShapedTypedFactory(Shape(axisExtent1, axisExtent2, axisExtent3), vtype)
