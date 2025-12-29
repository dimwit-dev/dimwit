package dimwit.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import dimwit.jax.Jax
import dimwit.jax.JaxDType
import dimwit.jax.Jax.PyDynamic
import dimwit.tensor.{Label, Labels, ExecutionType, VType}
//import dimwit.random.Random
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random
import me.shadaj.scalapy.readwrite.Writer
import scala.reflect.ClassTag
import scala.annotation.unchecked.uncheckedVariance

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])
  case Other(pyDevice: PyDynamic) extends Device(pyDevice)

object Device:
  val default: Device = Device.CPU
  val values: Seq[Device] = Seq(
    Device.CPU
  )

class Tensor[+T <: Tuple: Labels, V] private[tensor] (
    val jaxValue: Jax.PyDynamic
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)
  lazy val vtype: VType[V] = VType(this)

  lazy val device: Device = Device.values.find(d => Jax.device_get(jaxValue).equals(d.jaxDevice)).getOrElse(Device.Other(Jax.device_get(jaxValue)))

  def asType[V2](vtype: VType[V2]): Tensor[T, V2] = new Tensor(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(vtype.dtype)))

  def toDevice(newDevice: Device): Tensor[T, V] = new Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  override def equals(other: Any): Boolean =
    other match
      case that: Tensor[?, ?] => Jax.jnp.array_equal(this.jaxValue, that.jaxValue).item().as[Boolean]
      case _                  => false

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Dim[L] =
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def randn[T <: Tuple: Labels](shape: Shape[T])(key: Random.Key)(using
      executionType: ExecutionType[Float]
  ): Tensor[T, Float] = Random.Normal(shape)(key)

  def fromPy[T <: Tuple: Labels, V](vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def zeros[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): Tensor[T, V] = Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = vtype.dtype.jaxType))
  def ones[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): Tensor[T, V] = Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = vtype.dtype.jaxType))
  def const[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V])(value: V)(using writer: Writer[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))
  def fromArray[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V])(values: Array[V])(using
      py.ConvertableToSeqElem[V]
  ): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = vtype.dtype.jaxType
      )
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  given Conversion[Float, Tensor0[Float]] = (x: Float) => Tensor0(x)
  given Conversion[Int, Tensor0[Int]] = (x: Int) => Tensor0(x)
  given Conversion[Boolean, Tensor0[Boolean]] = (x: Boolean) => Tensor0(x)

  def zero[V](vtype: VType[V]): Tensor0[V] = Tensor.zeros(Shape.empty, vtype)
  def one[V](vtype: VType[V]): Tensor0[V] = Tensor.ones(Shape.empty, vtype)
  def const[V](vtype: VType[V])(value: V)(using writer: Writer[V]): Tensor0[V] = Tensor.const(Shape.empty, vtype)(value)

  def randn(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor0[Float] = Random.Normal(Shape.empty)(key)
  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)
  def apply[V](value: V)(using sv: ExecutionType[V], writer: Writer[V]): Tensor0[V] = Tensor0.const(VType[V])(value)

object Tensor1:

  def fromArray[L: Label, V](axis: Axis[L], vtype: VType[V])(values: Array[V])(using
      py.ConvertableToSeqElem[V]
  ): Tensor1[L, V] = Tensor(
    Jax.jnp.array(
      values.toPythonProxy,
      dtype = vtype.dtype.jaxType
    )
  )

object Tensor2:

  def fromArray[L1: Label, L2: Label, V](
      axis1: Axis[L1],
      axis2: Axis[L2],
      vtype: VType[V]
  )(
      values: Array[Array[V]]
  )(using
      py.ConvertableToSeqElem[V],
      ClassTag[V]
  ): Tensor2[L1, L2, V] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten)

  def eye[L: Label, V](dim: Dim[L], vtype: VType[V]): Tensor2[L, L, V] = Tensor(Jax.jnp.eye(dim._2, dtype = vtype.dtype.jaxType))
  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, L, V] = Tensor(Jax.jnp.diag(diag.jaxValue))
