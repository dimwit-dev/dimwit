package shapeful.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jax.Jax
import shapeful.jax.JaxDType
import shapeful.jax.Jax.PyDynamic
import shapeful.tensor.{Label, Labels}
//import shapeful.random.Random
import me.shadaj.scalapy.py.SeqConverters

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  // case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])
  case Other(name: String) extends Device(Jax.Dynamic.global.none)

object Device:
  val default: Device = Device.CPU
  val values: Seq[Device] = Seq(
    Device.CPU
  )

class Tensor[T <: Tuple : Labels, V : Value] private[tensor](
  val jaxValue: Jax.PyDynamic,
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values.find(
    d => Jax.device_get(jaxValue).equals(d.jaxDevice)
  ).getOrElse(Device.Other(Jax.device_get(jaxValue).name.as[String]))

  def asType[V2 : Value]: Tensor[T, V2] = 
    Tensor[T, V2](jaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(summon[Value[V2]].dtype)))

  def toDevice(newDevice: Device): Tensor[T, V] = 
    Tensor[T, V](jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  def equals(other: Tensor[T, V]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T, L]): Dim[L] = 
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [ _ ] =>> Int]

  private[tensor] def apply[T <: Tuple : Labels, V : Value](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor[T, V](jaxValue)

  def fromPy[T <: Tuple : Labels, V : Value](jaxValue: Jax.PyDynamic): Tensor[T, V] = Tensor[T, V](jaxValue)

  // Builder pattern for type-safe tensor creation
  class TensorBuilder[V : Value]:
    private val dtype = summon[Value[V]].dtype

    def apply[T <: Tuple : Labels](shape: Shape[T], values: Array[Float], device: Device = Device.default): Tensor[T, V] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val jaxValues = Jax.jnp
        .array(
          values.toPythonProxy,
          dtype = dtype.jaxType,
          device = device.jaxDevice,
        )
        .reshape(shape.dimensions.toPythonProxy)
      Tensor[T, V](jaxValues)

    def zeros[T <: Tuple : Labels](shape: Shape[T]): Tensor[T, V] =
      Tensor[T, V](Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

    def ones[T <: Tuple : Labels](shape: Shape[T]): Tensor[T, V] =
      Tensor[T, V](Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

  def of[V : Value]: TensorBuilder[V] = new TensorBuilder[V]

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  class Tensor0Builder[V : Value]:
    private val dtype = summon[Value[V]].dtype

    def apply(value: Float): Tensor0[V] =
      Tensor0[V](Jax.jnp.array(value, dtype=dtype.jaxType))

    def apply(value: Int): Tensor0[V] =
      Tensor0[V](Jax.jnp.array(value, dtype=dtype.jaxType))

    def apply(value: Boolean): Tensor0[V] =
      Tensor0[V](Jax.jnp.array(value, dtype=dtype.jaxType))

  def of[V : Value]: Tensor0Builder[V] = new Tensor0Builder[V]

  private[tensor] def apply[V : Value](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor[EmptyTuple, V](jaxValue)

object Tensor1:

  class Tensor1Builder[V : Value]:
    private val dtype = summon[Value[V]].dtype

    def apply[L : Label](axis: Axis[L], values: Array[Float]): Tensor1[L, V] =
      Tensor[Tuple1[L], V](Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

    def fromInts[L : Label](axis: Axis[L], values: Array[Int]): Tensor1[L, V] =
      Tensor[Tuple1[L], V](Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

  def of[V : Value]: Tensor1Builder[V] = new Tensor1Builder[V]

object Tensor2:

  class Tensor2Builder[V : Value]:
    private val dtype = summon[Value[V]].dtype

    def apply[L1 : Label, L2 : Label](
        shape: Shape2[L1, L2],
        values: Array[Float],
    ): Tensor2[L1, L2, V] = 
      Tensor.of[V].apply(shape, values)

    def apply[L1 : Label, L2 : Label](
        axis1: Axis[L1],
        axis2: Axis[L2],
        values: Array[Array[Float]],
    ): Tensor2[L1, L2, V] =
      val rows = values.length
      val cols = values.headOption.map(_.length).getOrElse(0)
      require(values.forall(_.length == cols), "All rows must have the same length")
      Tensor2.of[V].apply(Shape(axis1 -> rows, axis2 -> cols), values.flatten)

    def eye[L : Label](axis: Axis[L])(dim: Int): Tensor2[L, L, V] = 
      Tensor[Tuple2[L, L], V](Jax.jnp.eye(dim, dtype = dtype.jaxType))

    def diag[L : Label](diag: Tensor1[L, V]): Tensor2[L, L, V] =
      Tensor[Tuple2[L, L], V](Jax.jnp.diag(diag.jaxValue))

  def of[V : Value]: Tensor2Builder[V] = new Tensor2Builder[V]

object Tensor3:

  class Tensor3Builder[V : Value]:
    private val dtype = summon[Value[V]].dtype

    def apply[L1 : Label, L2 : Label, L3 : Label](
        shape: Shape3[L1, L2, L3],
        values: Array[Float],
    ): Tensor3[L1, L2, L3, V] = 
      Tensor.of[V].apply(shape, values)

    def apply[L1 : Label, L2 : Label, L3 : Label](
        axis1: Axis[L1],
        axis2: Axis[L2],
        axis3: Axis[L3],
        values: Array[Array[Array[Float]]],
    ): Tensor3[L1, L2, L3, V] =
      val dim1 = values.length
      val dim2 = values.headOption.map(_.length).getOrElse(0)
      val dim3 = values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0)
      require(values.forall(_.length == dim2), "All second dimensions must match")
      require(values.forall(_.forall(_.length == dim3)), "All third dimensions must match")
      Tensor3.of[V].apply(
        Shape(axis1 -> dim1, axis2 -> dim2, axis3 -> dim3),
        values.flatten.flatten,
      )

  def of[V : Value]: Tensor3Builder[V] = new Tensor3Builder[V]
