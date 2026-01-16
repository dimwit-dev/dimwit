package dimwit.tensor
import dimwit.jax.JaxDType
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum DType(val name: String, val size: Int):
  case Float32 extends DType("float32", 4)
  case Float64 extends DType("float64", 8)
  case Int32 extends DType("int32", 4)
  case Int64 extends DType("int64", 8)
  case Int16 extends DType("int16", 2)
  case Int8 extends DType("int8", 1)
  case UInt32 extends DType("uint32", 4)
  case UInt16 extends DType("uint16", 2)
  case UInt8 extends DType("uint8", 1)
  case Bool extends DType("bool", 1)
  case Complex64 extends DType("complex64", 8)
  case Complex128 extends DType("complex128", 16)

  lazy val jaxType = JaxDType.jaxDtype(this)

  /** Writes the scala array into the buffer according to THIS DType's format.
    * Handles conversions (e.g. Array[Double] -> Float32 buffer).
    */
  def write(values: Array[?]): Array[Byte] =
    val buffer = ByteBuffer.allocate(values.length * this.size)
    buffer.order(ByteOrder.LITTLE_ENDIAN) // Python uses little-endian

    // write values into buffer according to this DType
    (this, values) match
      // --- Float32 Target ---
      case (Float32, arr: Array[Float]) =>
        buffer.asFloatBuffer().put(arr)
      case (Float32, arr: Array[Double]) =>
        val fb = buffer.asFloatBuffer()
        var i = 0
        while i < arr.length do
          fb.put(arr(i).toFloat); i += 1

      // --- Float64 Target ---
      case (Float64, arr: Array[Double]) =>
        buffer.asDoubleBuffer().put(arr)
      case (Float64, arr: Array[Float]) =>
        val db = buffer.asDoubleBuffer()
        var i = 0;
        while i < arr.length do
          db.put(arr(i).toDouble); i += 1

      // --- Int32 Target ---
      case (Int32, arr: Array[Int]) =>
        buffer.asIntBuffer().put(arr)
      case (Int32, arr: Array[Byte]) =>
        val ib = buffer.asIntBuffer()
        var i = 0;
        while i < arr.length do
          ib.put(arr(i).toInt); i += 1

      // --- Int8 Target ---
      case (Int8, arr: Array[Byte]) =>
        buffer.put(arr)

      case (UInt8, arr: Array[Byte]) =>
        buffer.put(arr) // Interpret bytes as unsigned when reading

      // --- Bool Target (1 byte per bool) ---
      case (Bool, arr: Array[Boolean]) =>
        var i = 0
        while i < arr.length do
          buffer.put(if arr(i) then 1.toByte else 0.toByte)
          i += 1

      // --- Fallback/Error ---
      case _ =>
        throw new IllegalArgumentException(s"Conversion from ${values.getClass.getSimpleName} to DType $name is not supported or implemented.")

    buffer.array()
