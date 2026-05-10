package dimwit.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import dimwit.jax.Jax
import dimwit.jax.JaxDType
import dimwit.jax.Jax.PyDynamic
import dimwit.tensor.{Label, Labels, ExecutionType, VType}
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
import dimwit.tensor.TensorOps.IsFloating
import DType.*
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TensorOps.IsBoolean

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

  case class Factory[T <: Tuple: Labels](override val shape: Shape[T]) extends FactoryLowPrio[T]:

    def fill(value: Float): Tensor[T, Float32] = fill[Float32](value)
    def fill(value: Double): Tensor[T, Float64] = fill[Float64](value)
    def fromArray(values: Array[Float]): Tensor[T, Float32] = fromArray[Float32](values)
    def fromArray(values: Array[Double]): Tensor[T, Float64] = fromArray[Float64](values)

    def fill(value: Byte): Tensor[T, Int8] = fill[Int8](value)
    def fill(value: Short): Tensor[T, Int16] = fill[Int16](value)
    def fill(value: Int): Tensor[T, Int32] = fill[Int32](value)
    def fill(value: Long): Tensor[T, Int64] = fill[Int64](value)
    def fromArray(values: Array[Byte]): Tensor[T, Int8] = fromArray[Int8](values)
    def fromArray(values: Array[Short]): Tensor[T, Int16] = fromArray[Int16](values)
    def fromArray(values: Array[Int]): Tensor[T, Int32] = fromArray[Int32](values)
    def fromArray(values: Array[Long]): Tensor[T, Int64] = fromArray[Int64](values)

    def fill(value: Boolean): Tensor[T, Bool] = fill[Bool](value)
    def fromArray(values: Array[Boolean]): Tensor[T, Bool] = fromArray[Bool](values)

  trait FactoryLowPrio[T <: Tuple: Labels]:

    def shape: Shape[T]

    def fill[V: IsBoolean](value: Boolean): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsBoolean[V].dtype.jaxType))
    def fromArray[V: IsBoolean](values: Array[Boolean]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

    def fill[V: IsInteger](value: Byte): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))
    def fill[V: IsInteger](value: Short): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value.toInt, dtype = IsInteger[V].dtype.jaxType))
    def fill[V: IsInteger](value: Int): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))
    def fill[V: IsInteger](value: Long): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))
    def fromArray[V: IsInteger](values: Array[Byte]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray[V: IsInteger](values: Array[Short]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray[V: IsInteger](values: Array[Int]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray[V: IsInteger](values: Array[Long]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

    def fill[V: IsFloating](value: Float): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsFloating[V].dtype.jaxType))
    def fill[V: IsFloating](value: Double): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = IsFloating[V].dtype.jaxType))
    def fromArray[V: IsFloating](values: Array[Float]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)
    def fromArray[V: IsFloating](values: Array[Double]): Tensor[T, V] = ArrayWriter.fromArray[T, V](shape, values)

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

  private[dimwit] def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def apply[T <: Tuple: Labels](shape: Shape[T]): Tensor.Factory[T] = Tensor.Factory(shape)

  def like[T <: Tuple: Labels, V](template: Tensor[T, V]): Tensor.LikeFactory[T, V] = Tensor.LikeFactory(template)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0 extends Tensor0LowPrio:

  def apply(value: Boolean): Tensor0[Bool] = apply[Bool](value)

  def apply(value: Byte): Tensor0[Int8] = apply[Int8](value)
  def apply(value: Short): Tensor0[Int16] = apply[Int16](value)
  def apply(value: Int): Tensor0[Int32] = apply[Int32](value)
  def apply(value: Long): Tensor0[Int64] = apply[Int64](value)

  def apply(value: Float): Tensor0[Float32] = apply[Float32](value)
  def apply(value: Double): Tensor0[Float64] = apply[Float64](value)

trait Tensor0LowPrio:

  def apply[V: IsBoolean](value: Boolean): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsBoolean[V].dtype.jaxType))

  def apply[V: IsInteger](value: Byte): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))
  def apply[V: IsInteger](value: Short): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value.toInt, dtype = IsInteger[V].dtype.jaxType))
  def apply[V: IsInteger](value: Int): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))
  def apply[V: IsInteger](value: Long): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsInteger[V].dtype.jaxType))

  def apply[V: IsFloating](value: Float): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsFloating[V].dtype.jaxType))
  def apply[V: IsFloating](value: Double): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = IsFloating[V].dtype.jaxType))

  def like[V: Writer](template: Tensor0[V])(value: V): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))

  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)

object Tensor1:

  case class Factory[L: Label]() extends FactoryLowPrio[L]:

    def fromArray(values: Array[Boolean]): Tensor1[L, Bool] = fromArray[Bool](values)

    def fromArray(values: Array[Byte]): Tensor1[L, Int8] = fromArray[Int8](values)
    def fromArray(values: Array[Short]): Tensor1[L, Int16] = fromArray[Int16](values)
    def fromArray(values: Array[Int]): Tensor1[L, Int32] = fromArray[Int32](values)
    def fromArray(values: Array[Long]): Tensor1[L, Int64] = fromArray[Int64](values)

    def fromArray(values: Array[Float]): Tensor1[L, Float32] = fromArray[Float32](values)
    def fromArray(values: Array[Double]): Tensor1[L, Float64] = fromArray[Float64](values)

  trait FactoryLowPrio[L: Label]:

    def fromArray[V: IsBoolean](values: Array[Boolean]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)

    def fromArray[V: IsInteger](values: Array[Byte]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)
    def fromArray[V: IsInteger](values: Array[Short]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)
    def fromArray[V: IsInteger](values: Array[Int]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)
    def fromArray[V: IsInteger](values: Array[Long]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)

    def fromArray[V: IsFloating](values: Array[Float]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)
    def fromArray[V: IsFloating](values: Array[Double]): Tensor1[L, V] = ArrayWriter.fromArray[Tuple1[L], V](Shape1(Axis[L] -> values.length), values)

  def apply[L: Label](axis: Axis[L]): Factory[L] = Factory()

object Tensor2:

  type Array2D[V] = Array[Array[V]]

  case class Factory[L: Label]() extends FactoryLowPrio[L]:

    def fromArray(values: Array2D[Boolean]): Tensor1[L, Bool] = fromArray[Bool](values)

    def fromArray(values: Array2D[Byte]): Tensor1[L, Int8] = fromArray[Int8](values)
    def fromArray(values: Array2D[Short]): Tensor1[L, Int16] = fromArray[Int16](values)
    def fromArray(values: Array2D[Int]): Tensor1[L, Int32] = fromArray[Int32](values)
    def fromArray(values: Array2D[Long]): Tensor1[L, Int64] = fromArray[Int64](values)

    def fromArray(values: Array2D[Float]): Tensor1[L, Float32] = fromArray[Float32](values)
    def fromArray(values: Array2D[Double]): Tensor1[L, Float64] = fromArray[Float64](values)

  trait FactoryLowPrio[L: Label]:

    def fromArray[V: IsBoolean](values: Array2D[Boolean]): Tensor1[L, V] = ???

    def fromArray[V: IsInteger](values: Array2D[Byte]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array2D[Short]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array2D[Int]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array2D[Long]): Tensor1[L, V] = ???

    def fromArray[V: IsFloating](values: Array2D[Float]): Tensor1[L, V] = ???
    def fromArray[V: IsFloating](values: Array2D[Double]): Tensor1[L, V] = ???

  def apply[L: Label](axis: Axis[L]): Factory[L] = Factory()

  private def eyeImpl[L: Label, V](dim: AxisExtent[L], dtype: DType): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.eye(dim.size, dtype = dtype.jaxType))
  def eye[L: Label](dim: AxisExtent[L])(using et: ExecutionType[Float]): Tensor2[L, Prime[L], Float] = eyeImpl(dim, et.dtype)
  def eye[L: Label, V](dim: AxisExtent[L], vtype: VType[V]): Tensor2[L, Prime[L], V] = eyeImpl(dim, vtype.dtype)
  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  type Array3D[V] = Array[Array[Array[V]]]

  case class Factory[L: Label]() extends FactoryLowPrio[L]:

    def fromArray(values: Array3D[Boolean]): Tensor1[L, Bool] = fromArray[Bool](values)

    def fromArray(values: Array3D[Byte]): Tensor1[L, Int8] = fromArray[Int8](values)
    def fromArray(values: Array3D[Short]): Tensor1[L, Int16] = fromArray[Int16](values)
    def fromArray(values: Array3D[Int]): Tensor1[L, Int32] = fromArray[Int32](values)
    def fromArray(values: Array3D[Long]): Tensor1[L, Int64] = fromArray[Int64](values)

    def fromArray(values: Array3D[Float]): Tensor1[L, Float32] = fromArray[Float32](values)
    def fromArray(values: Array3D[Double]): Tensor1[L, Float64] = fromArray[Float64](values)

  trait FactoryLowPrio[L: Label]:

    def fromArray[V: IsBoolean](values: Array3D[Boolean]): Tensor1[L, V] = ???

    def fromArray[V: IsInteger](values: Array3D[Byte]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array3D[Short]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array3D[Int]): Tensor1[L, V] = ???
    def fromArray[V: IsInteger](values: Array3D[Long]): Tensor1[L, V] = ???

    def fromArray[V: IsFloating](values: Array3D[Float]): Tensor1[L, V] = ???
    def fromArray[V: IsFloating](values: Array3D[Double]): Tensor1[L, V] = ???

  def apply[L: Label](axis: Axis[L]): Factory[L] = Factory()
