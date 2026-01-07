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
