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

enum Device(val platform: String):
  case CPU extends Device("cpu")
  case GPU extends Device("gpu")
  case Other extends Device("other")

object Device:
  val default: Device = Device.CPU
  extension (device: Device)
    def toJaxDevice: Jax.PyDynamic =
      val devices = Jax.devices(device.platform)
      require(devices.nonEmpty, s"No JAX devices found for platform: ${device.platform}")
      devices.head

class Tensor[T <: Tuple: Labels, V] private[tensor] (
    val jaxValue: Jax.PyDynamic
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromSeq[T](jaxValue.shape.as[Seq[Int]])
  lazy val vtype: VType[V] = VType(this)

  lazy val device: Device =
    val jaxDevice = Jax.device_get(jaxValue)
    jaxDevice.platform.as[String] match
      case "cpu" => Device.CPU
      case "gpu" => Device.GPU
      case _     => Device.Other

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

  def extent[L](axis: Axis[L])(using ev: AxisIndex[T, L]): AxisExtent[L] =
    shape.extent(axis)

  private val jaxTypeName: String = py.Dynamic.global.`type`(jaxValue).`__name__`.as[String]
  lazy val isTracer: Boolean =
    val jaxCoreTracer = py.module("jax.core").Tracer
    py.Dynamic.global.isinstance(jaxValue, jaxCoreTracer).as[Boolean]

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  case class Factory[T <: Tuple: Labels](val shape: Shape[T]):

    def fill[A: ExecutionType: Writer, V](value: A)(using ev: WriterEvidence.Aux[A, V]): Tensor[T, V] =
      Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = ExecutionType[A].dtype.jaxType))

    def fromArray[A: ExecutionType, V](values: Array[A])(using t2a: ArrayWriter.Aux[A, V]): Tensor[T, V] =
      t2a.fromArray[T](shape)(values)

  case class LikeFactory[T <: Tuple: Labels, V](val other: Tensor[T, V]):

    def fill[A: Writer](value: A)(using ev: WriterEvidence.Aux[A, V]): Tensor[T, V] =
      Tensor(Jax.jnp.full(other.shape.dimensions.toPythonProxy, value, dtype = other.dtype.jaxType))

    def fromArray[A](values: Array[A])(using t2a: ArrayWriter.Aux[A, V]): Tensor[T, V] =
      given ExecutionType[A] = ExecutionTypeFor[A](other.dtype) // fix the underlying dtype to match the other tensor's dtype
      summon[ArrayWriter[A]].fromArray[T](other.shape)(values)

  def apply[T <: Tuple: Labels](shape: Shape[T]): Tensor.Factory[T] = Tensor.Factory(shape)
  def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def like[T <: Tuple: Labels, V](template: Tensor[T, V]): Tensor.LikeFactory[T, V] = Tensor.LikeFactory(template)

  def fromPy[T <: Tuple: Labels, V](vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  given float2FloatTensor: Conversion[Float, Tensor0[Float]] = (x: Float) => Tensor0(x)
  given int2IntTensor: Conversion[Int, Tensor0[Int]] = (x: Int) => Tensor0(x)
  given int2FloatTensor: Conversion[Int, Tensor0[Float]] = (x: Int) => Tensor0(x.toFloat)
  given boolean2BooleanTensor: Conversion[Boolean, Tensor0[Boolean]] = (x: Boolean) => Tensor0(x)

  def apply[V: ExecutionType: Writer](value: V): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = ExecutionType[V].dtype.jaxType))
  def like[V: Writer](template: Tensor0[V])(value: V): Tensor0[V] = Tensor(Jax.jnp.full(Shape0.dimensions.toPythonProxy, value, dtype = template.dtype.jaxType))

  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)

object Tensor1:

  case class Factory[L: Label](val axis: Axis[L]):
    private def createShape(l: Int): Shape1[L] = Shape1(AxisExtent(axis, l))
    def fromArray[A: ExecutionType, V](values: Array[A])(using t2a: ArrayWriter.Aux[A, V]): Tensor[Tuple1[L], V] = Tensor(createShape(values.length)).fromArray(values)

  def apply[L: Label](axis: Axis[L]): Tensor1.Factory[L] = Tensor1.Factory(axis)

object Tensor2:

  case class Factory[L1: Label, L2: Label](val axis1: Axis[L1], val axis2: Axis[L2]):
    private def createShape[V](values: Array[Array[V]]): Shape2[L1, L2] = Shape2(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length))
    def fromArray[A: ClassTag: ExecutionType, V](values: Array[Array[A]])(using t2a: ArrayWriter.Aux[A, V]): Tensor[(L1, L2), V] = Tensor(createShape(values)).fromArray(values.flatten)

  def apply[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2.Factory[L1, L2] = Tensor2.Factory(axis1, axis2)

  private def eyeImpl[L: Label, V](dim: AxisExtent[L], dtype: DType): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.eye(dim.size, dtype = dtype.jaxType))
  def eye[L: Label](dim: AxisExtent[L])(using et: ExecutionType[Float]): Tensor2[L, Prime[L], Float] = eyeImpl(dim, et.dtype)
  def eye[L: Label, V](dim: AxisExtent[L], vtype: VType[V]): Tensor2[L, Prime[L], V] = eyeImpl(dim, vtype.dtype)
  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, Prime[L], V] = Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  case class Factory[L1: Label, L2: Label, L3: Label](val axis1: Axis[L1], val axis2: Axis[L2], val axis3: Axis[L3]):
    private def createShape[V](values: Array[Array[Array[V]]]): Shape3[L1, L2, L3] =
      Shape3(AxisExtent(axis1, values.length), AxisExtent(axis2, values.head.length), AxisExtent(axis3, values.head.head.length))
    def fromArray[A: ExecutionType: ClassTag, V](values: Array[Array[Array[A]]])(using t2a: ArrayWriter.Aux[A, V]): Tensor3[L1, L2, L3, V] =
      Tensor(createShape(values)).fromArray(values.flatten.flatten)

  def apply[L1: Label, L2: Label, L3: Label](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3]): Tensor3.Factory[L1, L2, L3] = Tensor3.Factory(axis1, axis2, axis3)
