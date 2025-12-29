package dimwit.tensor
import dimwit.jax.JaxDType

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

  // Default values with correct types
  def zero: Float | Double | Int | Long | Short | Byte | Boolean = this match
    case Float32    => 0.0f
    case Float64    => 0.0
    case Int32      => 0
    case Int64      => 0L
    case Int16      => 0.toShort
    case Int8       => 0.toByte
    case UInt32     => 0
    case UInt16     => 0.toShort
    case UInt8      => 0.toByte
    case Bool       => false
    case Complex64  => 0.0f // Real part of complex
    case Complex128 => 0.0

  // Type checking
  def isFloating: Boolean = this match
    case Float32 | Float64 => true
    case _                 => false

  def isInteger: Boolean = this match
    case Int32 | Int64 | Int16 | Int8 | UInt32 | UInt16 | UInt8 => true
    case _                                                      => false

  def isComplex: Boolean = this match
    case Complex64 | Complex128 => true
    case _                      => false

  def isUnsigned: Boolean = this match
    case UInt8 | UInt16 | UInt32 | Bool => true
    case _                              => false

  def isSigned: Boolean = this match
    case Bool | UInt8 | UInt16 | UInt32 => false
    case _                              => true

  def elementType: String = this match
    case Float32 | Float64            => "float"
    case Int8 | Int16 | Int32 | Int64 => "int"
    case UInt8 | UInt16 | UInt32      => "uint"
    case Bool                         => "bool"
    case Complex64 | Complex128       => "complex"

  // Get the corresponding unsigned type (if exists)
  def toUnsigned: Option[DType] = this match
    case Int8  => Some(UInt8)
    case Int16 => Some(UInt16)
    case Int32 => Some(UInt32)
    case _     => None

  // Get the corresponding signed type
  def toSigned: Option[DType] = this match
    case UInt8  => Some(Int8)
    case UInt16 => Some(Int16)
    case UInt32 => Some(Int32)
    case _      => None

  // Bit width for comparison
  def bitWidth: Int = size * 8

object DType:
  // Improved type promotion rules (following NumPy hierarchy)
  def promoteTypes(a: DType, b: DType): DType = (a, b) match
    // Same types
    case (a, b) if a == b => a

    // Bool promotes to anything
    case (Bool, other) => other
    case (other, Bool) => other

    // Complex types have highest precedence
    case (Complex128, _) | (_, Complex128) => Complex128
    case (Complex64, _) | (_, Complex64)   => Complex64

    // Float64 has highest precedence among real numbers
    case (Float64, _) | (_, Float64) => Float64
    case (Float32, _) | (_, Float32) => Float32

    // Integer promotion by signedness and bit width
    case (Int64, _) | (_, Int64)           => Int64
    case (UInt32, Int32) | (Int32, UInt32) => Int64 // Need wider type
    case (Int32, Int32)                    => Int32
    case (UInt32, UInt32)                  => UInt32
    case (Int16, _) | (_, Int16)           => Int32 // Promote smaller ints to Int32
    case (Int8, _) | (_, Int8)             => Int32
    case (UInt16, _) | (_, UInt16)         => Int32
    case (UInt8, _) | (_, UInt8)           => Int32

    // Fallback: choose the wider type
    case _ =>
      if a.bitWidth >= b.bitWidth then a else b

  // Check if promotion is safe (no precision loss)
  def canPromoteSafely(from: DType, to: DType): Boolean = (from, to) match
    case (a, b) if a == b                                 => true
    case (Bool, _)                                        => true
    case (Int8, Int16 | Int32 | Int64)                    => true
    case (Int16, Int32 | Int64)                           => true
    case (Int32, Int64)                                   => true
    case (UInt8, UInt16 | UInt32 | Int16 | Int32 | Int64) => true
    case (UInt16, UInt32 | Int32 | Int64)                 => true
    case (UInt32, Int64)                                  => true
    case (Int32 | Int64, Float32 | Float64)               => true
    case (Float32, Float64)                               => true
    case (Float32 | Float64, Complex64 | Complex128)      => true
    case (Complex64, Complex128)                          => true
    case _                                                => false

  // String parsing
  def fromString(s: String): Option[DType] = s.toLowerCase match
    case "float32" | "f32"            => Some(Float32)
    case "float64" | "f64" | "double" => Some(Float64)
    case "int32" | "i32" | "int"      => Some(Int32)
    case "int64" | "i64" | "long"     => Some(Int64)
    case "int16" | "i16" | "short"    => Some(Int16)
    case "int8" | "i8" | "byte"       => Some(Int8)
    case "uint32" | "u32"             => Some(UInt32)
    case "uint16" | "u16"             => Some(UInt16)
    case "uint8" | "u8"               => Some(UInt8)
    case "bool" | "boolean"           => Some(Bool)
    case "complex64" | "c64"          => Some(Complex64)
    case "complex128" | "c128"        => Some(Complex128)
    case _                            => None

  // More specific given conversions
  given floatToDType: Conversion[Float, DType] = _ => Float32
  given doubleToDType: Conversion[Double, DType] = _ => Float64
  given intToDType: Conversion[Int, DType] = _ => Int32
  given longToDType: Conversion[Long, DType] = _ => Int64
  given shortToDType: Conversion[Short, DType] = _ => Int16
  given byteToDType: Conversion[Byte, DType] = _ => Int8
  given boolToDType: Conversion[Boolean, DType] = _ => Bool
