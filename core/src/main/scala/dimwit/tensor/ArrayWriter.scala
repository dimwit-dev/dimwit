package dimwit.tensor

import java.nio.ByteBuffer
import java.util.Base64
import java.nio.ByteOrder
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer
import me.shadaj.scalapy.interpreter.PyValue
import dimwit.jax.Jax

trait WriterEvidence[A]:
  type V

/** Type class for providing evidence that a scalar of type A can be converted to type V using a ScalaPy Writer in a Tensor context.
  * The type A is the input scalar type, allowing to define an internal precision (dtype) based on the scalar type.
  * For example creating a Tensor[?, Int] from a scalar of type Byte with internal dtype uint8, int8, int16 or int32 (based on given ExecutionType[Byte]).
  *
  * The type V is the value type of the resulting Tensor (should be Boolean, Int or Float; or custom opaque types).
  */
object WriterEvidence:

  type Aux[A, V0] = WriterEvidence[A] { type V = V0 }

// Helper to instantiate
  def apply[A, V0]: Aux[A, V0] = new WriterEvidence[A]:
    type V = V0

  given Aux[Float, Float] = apply
  given Aux[Int, Int] = apply
  given Aux[Boolean, Boolean] = apply
  given Aux[Double, Float] = apply // Double casts to Float
  given Aux[Byte, Int] = apply // Byte casts to Int

/** Type class for creating Tensors of different value types from arrays of different base types.
  * While allowing to define an internal precision (dtype) based on the array type.
  * For example creating a Tensor[?, Int] from an Array[Byte] with internal dtype uint8, int8, int16 or int32 (based on given ExecutionType[Byte]).
  *
  * @param A The base type of the input array.
  * @param V The value type of the resulting Tensor.
  */
trait ArrayWriter[A]:
  type V
  def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[A]): Tensor[T, V]

object ArrayWriter:
  type Aux[A, V0] = ArrayWriter[A] { type V = V0 }

  val base64Loader = py.eval("lambda b64, shape, dtype: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=dtype).reshape(shape))")

  private def byteArrayToTensor[T <: Tuple: Labels, V](shape: Shape[T], byteArray: Array[Byte], jaxDType: Jax.PyDynamic): Tensor[T, V] =
    val b64String = Base64.getEncoder.encodeToString(byteArray)
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, jaxDType))

  given (using ExecutionType[Double]): ArrayWriter.Aux[Double, Float] = new ArrayWriter[Double]:
    type V = Float
    def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Double]): Tensor[T, Float] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val dtype = ExecutionType[Double].dtype
      val byteArray = dtype.write(values)
      byteArrayToTensor(shape, byteArray, dtype.jaxType)

  given (using ExecutionType[Float]): ArrayWriter.Aux[Float, Float] = new ArrayWriter[Float]:
    type V = Float
    def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Float]): Tensor[T, Float] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val dtype = ExecutionType[Float].dtype
      val byteArray = dtype.write(values)
      byteArrayToTensor(shape, byteArray, dtype.jaxType)

  given (using ExecutionType[Int]): ArrayWriter.Aux[Int, Int] = new ArrayWriter[Int]:
    type V = Int
    def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Int]): Tensor[T, Int] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val dtype = ExecutionType[Int].dtype
      val byteArray = dtype.write(values)
      byteArrayToTensor(shape, byteArray, dtype.jaxType)

  given (using ExecutionType[Byte]): ArrayWriter.Aux[Byte, Int] = new ArrayWriter[Byte]:
    type V = Int
    def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Byte]): Tensor[T, Int] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val dtype = ExecutionType[Byte].dtype
      val byteArray = dtype.write(values)
      byteArrayToTensor(shape, byteArray, dtype.jaxType)

  given (using ExecutionType[Boolean]): ArrayWriter.Aux[Boolean, Boolean] = new ArrayWriter[Boolean]:
    type V = Boolean
    def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Boolean]): Tensor[T, Boolean] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val dtype = ExecutionType[Boolean].dtype
      val byteArray = dtype.write(values)
      byteArrayToTensor(shape, byteArray, dtype.jaxType)
