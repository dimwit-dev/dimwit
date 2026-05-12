package dimwit.tensor
import dimwit.jax.JaxDType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import dimwit.tensor.TensorOps.{IsFloating, IsInteger, IsBoolean}

object DType:

  type UInt8 = UInt8.type
  given uint8IsFloating: IsInteger[UInt8] with
    def dtype: DType = DType.UInt8

  type UInt16 = UInt16.type
  given uint16IsInteger: IsInteger[UInt16] with
    def dtype: DType = DType.UInt16

  type UInt32 = UInt32.type
  given uint32IsInteger: IsInteger[UInt32] with
    def dtype: DType = DType.UInt32

  type Int8 = Int8.type
  given int8IsInteger: IsInteger[Int8] with
    def dtype: DType = DType.Int8

  type Int16 = Int16.type
  given int16IsInteger: IsInteger[Int16] with
    def dtype: DType = DType.Int16

  type Int32 = Int32.type
  given int32IsInteger: IsInteger[Int32] with
    def dtype: DType = DType.Int32

  type Int64 = Int64.type
  given int64IsInteger: IsInteger[Int64] with
    def dtype: DType = DType.Int64

  type Float16 = Float16.type
  given float16IsFloating: IsFloating[Float16] with
    def dtype: DType = DType.Float16

  type BFloat16 = BFloat16.type
  given bfloat16IsFloating: IsFloating[BFloat16] with
    def dtype: DType = DType.BFloat16

  type Float32 = Float32.type
  given float32IsFloating: IsFloating[Float32] with
    def dtype: DType = DType.Float32

  type Float64 = Float64.type
  given float64IsFloating: IsFloating[Float64] with
    def dtype: DType = DType.Float64

  type Bool = Bool.type
  given boolIsBoolean: IsBoolean[Bool] with
    def dtype: DType = DType.Bool

enum DType(val name: String, val size: Int):
  case BFloat16 extends DType("bfloat16", 2)
  case Float16 extends DType("float16", 2)
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
      case (Float16, arr: Array[Float]) =>
        val sb = buffer.asShortBuffer()
        var i = 0
        while i < arr.length do
          sb.put(floatToFloat16(arr(i)))
          i += 1
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

  private def floatToFloat16(f: Float): Short =
    // TODO replace with java.lang.Float.floatToFloat16 when we can require Java 20+
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    var valBits = bits & 0x7fffffff

    if valBits >= 0x47800000 then
      // NaN or Infinity
      if (valBits & 0x7f800000) == 0x7f800000 then
        if (valBits & 0x007fffff) != 0 then
          return (sign | 0x7c00 | (valBits & 0x007fffff) >>> 13).toShort // NaN
        return (sign | 0x7c00).toShort // Infinity
      return (sign | 0x7bff).toShort // Overflow

    if valBits >= 0x38800000 then
      // Normalized number
      return (sign | valBits - 0x38000000 >>> 13).toShort

    if valBits < 0x33000000 then
      // Underflow to zero
      return sign.toShort

    // Denormalized number
    valBits = (valBits & 0x007fffff) | 0x00800000
    val shift = 113 - (bits >>> 23 & 0xff)
    valBits = if shift < 24 then valBits >>> shift else 0
    (sign | valBits).toShort
