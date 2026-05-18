package dimwit.tensor

import java.nio.ByteBuffer
import java.util.Base64
import java.nio.ByteOrder
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer
import me.shadaj.scalapy.interpreter.PyValue
import dimwit.jax.Jax
import dimwit.tensor.TensorOps.{IsBoolean, IsInteger, IsFloating}

object ArrayWriter:

  val base64Loader = py.eval("lambda b64, shape, dtype: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=dtype).reshape(shape))")

  private def byteArrayToTensor[T <: Tuple: Labels, V](shape: Shape[T], byteArray: Array[Byte], jaxDType: Jax.PyDynamic): Tensor[T, V] =
    val b64String = Base64.getEncoder.encodeToString(byteArray)
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, jaxDType))

  def fromArray[T <: Tuple: Labels, V: IsFloating](shape: Shape[T], values: Array[Double]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsFloating[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsFloating](shape: Shape[T], values: Array[Float]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsFloating[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsInteger](shape: Shape[T], values: Array[Int]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsInteger[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsInteger](shape: Shape[T], values: Array[Long]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsInteger[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsInteger](shape: Shape[T], values: Array[Byte]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsInteger[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsInteger](shape: Shape[T], values: Array[Short]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsInteger[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)

  def fromArray[T <: Tuple: Labels, V: IsBoolean](shape: Shape[T], values: Array[Boolean]): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val dtype = IsBoolean[V].dtype
    byteArrayToTensor(shape, dtype.write(values), dtype.jaxType)
