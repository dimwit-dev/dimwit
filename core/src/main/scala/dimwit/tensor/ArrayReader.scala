package dimwit.tensor

import java.nio.ByteBuffer
import java.nio.ByteOrder
import me.shadaj.scalapy.py

/** Utility object for reading flat arrays of scalar values from JAX tensors.
  */
object ArrayReader:

  private def readBytes(jaxValue: py.Dynamic): Array[Byte] =
    jaxValue.tobytes().as[Seq[Byte]].toArray

  def readBooleanArray(jaxValue: py.Dynamic): Array[Boolean] =
    val bytes = readBytes(jaxValue)
    Array.tabulate(bytes.length)(i => bytes(i) != 0)

  def readByteArray(jaxValue: py.Dynamic): Array[Byte] =
    readBytes(jaxValue)

  def readShortArray(jaxValue: py.Dynamic): Array[Short] =
    val bytes = readBytes(jaxValue)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    Array.tabulate(buf.remaining())(buf.get)

  def readIntArray(jaxValue: py.Dynamic): Array[Int] =
    val bytes = readBytes(jaxValue)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
    Array.tabulate(buf.remaining())(buf.get)

  def readLongArray(jaxValue: py.Dynamic): Array[Long] =
    val bytes = readBytes(jaxValue)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
    Array.tabulate(buf.remaining())(buf.get)

  def readFloatArray(jaxValue: py.Dynamic): Array[Float] =
    val bytes = readBytes(jaxValue)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    Array.tabulate(buf.remaining())(buf.get)

  def readDoubleArray(jaxValue: py.Dynamic): Array[Double] =
    val bytes = readBytes(jaxValue)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
    Array.tabulate(buf.remaining())(buf.get)
