package shapeful.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jax.Jax
import shapeful.jax.JaxDType
import shapeful.jax.Jax.PyDynamic
import shapeful.tensor.{Label, Labels, ExecutionType, Of}
//import shapeful.random.Random
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import shapeful.random.Random
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

class Tensor[+T <: Tuple : Labels, V] private[tensor](
  val jaxValue: Jax.PyDynamic,
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values.find(
    d => Jax.device_get(jaxValue).equals(d.jaxDevice)
  ).getOrElse(Device.Other(Jax.device_get(jaxValue)))

  def asType[V2](of: Of[V2]): Tensor[T, V2] = new Tensor(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(of.dtype)))

  def toDevice(newDevice: Device): Tensor[T, V] = new Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Dim[L] = 
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  def apply[T <: Tuple : Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def apply[V](tv: Of[V]) = new TensorFactory[V](tv)
  def randn[T <: Tuple: Labels](shape: Shape[T])(key: Random.Key)(using 
    executionType: ExecutionType[Float]
  ): Tensor[T, Float] = Random.Normal(key, shape)

  class TensorFactory[V](val tv: Of[V]):
    def fromPy[T <: Tuple: Labels](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
    def zeros[T <: Tuple: Labels](shape: Shape[T]): Tensor[T, V] = fromPy(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = tv.dtype.jaxType))
    def ones[T <: Tuple: Labels](shape: Shape[T]): Tensor[T, V] = fromPy(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = tv.dtype.jaxType))
    def const[T <: Tuple: Labels](value: V)(shape: Shape[T])(using writer: Writer[V]): Tensor[T, V] = Tensor(tv).fromPy(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = tv.dtype.jaxType))
    def fromArray[T <: Tuple: Labels](shape: Shape[T], values: Array[V])(using 
      py.ConvertableToSeqElem[V],
    ): Tensor[T, V] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val jaxValues = Jax.jnp.array(
        values.toPythonProxy,
        dtype = tv.dtype.jaxType
      ).reshape(shape.dimensions.toPythonProxy)
      fromPy(jaxValues)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  given Conversion[Float, Tensor0[Float]] = (x: Float) => Tensor0(x)
  given Conversion[Int, Tensor0[Int]] = (x: Int) => Tensor0(x)
  given Conversion[Boolean, Tensor0[Boolean]] = (x: Boolean) => Tensor0(x)

  class Tensor0Factory[V](tv: Of[V]):
    def zero: Tensor0[V] = Tensor(tv).zeros(Shape.empty)
    def one: Tensor0[V] = Tensor(tv).ones(Shape.empty)
    def const(value: V)(using writer: Writer[V]): Tensor0[V] = Tensor(tv).const(value)(Shape.empty)

  def randn(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor0[Float] = Random.Normal(key, Shape.empty)
  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)
  def apply[V](tv: Of[V]): Tensor0Factory[V] = Tensor0Factory[V](tv)
  def apply[V](value: V)(using sv: ExecutionType[V], writer: Writer[V]): Tensor0[V] = Tensor(Of[V]).const(value)(Shape.empty)

object Tensor1:

  class Tensor1Factory[V](val tv: Of[V]):
    def zeros[L: Label](dim: Dim[L]): Tensor1[L, V] = Tensor(tv).zeros(Shape1(dim))
    def ones[L: Label](dim: Dim[L]): Tensor1[L, V] = Tensor(tv).ones(Shape1(dim))
    def const[L: Label](value: V)(dim: Dim[L])(using writer: Writer[V]): Tensor1[L, V] = Tensor(tv).const(value)(Shape1(dim))
    def fromArray[L: Label](axis: Axis[L], values: Array[V])(
      using py.ConvertableToSeqElem[V]
    ): Tensor1[L, V] = Tensor(tv).fromPy(Jax.jnp.array(
      values.toPythonProxy,
      dtype = tv.dtype.jaxType
    ))

  def randn[L: Label](dim: Dim[L])(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor1[L, Float] = Random.Normal(key, Shape1(dim))
  def apply[V](tv: Of[V]): Tensor1Factory[V] = Tensor1Factory[V](tv)
  
object Tensor2:

  class Tensor2Factory[V](val tv: Of[V]):
    def zeros[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2]): Tensor2[L1, L2, V] = Tensor(tv).zeros(Shape2(dim1, dim2))
    def ones[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2]): Tensor2[L1, L2, V] = Tensor(tv).ones(Shape2(dim1, dim2))
    def const[L1: Label, L2: Label](value: V)(dim1: Dim[L1], dim2: Dim[L2])(using writer: Writer[V]): Tensor2[L1, L2, V] = Tensor(tv).const(value)(Shape2(dim1, dim2))
    def fromArray[L1: Label, L2: Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: Array[Array[V]]
    )(using 
      py.ConvertableToSeqElem[V],
      ClassTag[V],
    ): Tensor2[L1, L2, V] = 
      val dims = (axis1 -> values.length, axis2 -> values.head.length)
      Tensor(tv).fromArray(Shape(dims), values.flatten)

    def eye[L: Label](dim: Dim[L]): Tensor2[L, L, V] = Tensor(tv).fromPy(Jax.jnp.eye(dim._2, dtype = tv.dtype.jaxType))
    def diag[L: Label](diag: Tensor1[L, V]): Tensor2[L, L, V] = Tensor(tv).fromPy(Jax.jnp.diag(diag.jaxValue))

  def randn[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2])(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor2[L1, L2, Float] = Random.Normal(key, Shape2(dim1, dim2))
  def apply[V](tv: Of[V]): Tensor2Factory[V] = Tensor2Factory[V](tv)

object Tensor3:

  class Tensor3Factory[V](val tv: Of[V]):
    def zeros[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3]): Tensor3[L1, L2, L3, V] = Tensor(tv).zeros(Shape3(dim1, dim2, dim3))
    def ones[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3]): Tensor3[L1, L2, L3, V] = Tensor(tv).ones(Shape3(dim1, dim2, dim3))
    def const[L1: Label, L2: Label, L3: Label](value: V)(dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3])(using writer: Writer[V]): Tensor3[L1, L2, L3, V] = Tensor(tv).const(value)(Shape3(dim1, dim2, dim3))
  end Tensor3Factory

  def randn[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3])(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor3[L1, L2, L3, Float] = Random.Normal(key, Shape3(dim1, dim2, dim3))
  def apply[V](tv: Of[V]): Tensor3Factory[V] = Tensor3Factory[V](tv)
