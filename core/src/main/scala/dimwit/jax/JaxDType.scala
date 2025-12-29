package dimwit.jax

import me.shadaj.scalapy.py
import dimwit.tensor.DType

object JaxDType:

  // Create JAX array with specific dtype
  def createJaxArray(data: py.Any, dtype: DType): py.Dynamic =
    try
      val jnp = Jax.jnp
      jnp.array(data, dtype = jaxDtype(dtype))
    catch
      case e: Exception =>
        throw new RuntimeException(s"Failed to create JAX array with dtype ${dtype.name}: ${e.getMessage}", e)

  // Extract DType from JAX object
  def fromJaxDtype(jaxDtype: py.Dynamic): DType =
    try
      val dtypeStr = jaxDtype.name.as[String]
      dtypeStr match
        case "float32"    => DType.Float32
        case "float64"    => DType.Float64
        case "int32"      => DType.Int32
        case "int64"      => DType.Int64
        case "int16"      => DType.Int16
        case "int8"       => DType.Int8
        case "uint32"     => DType.UInt32
        case "uint16"     => DType.UInt16
        case "uint8"      => DType.UInt8
        case "bool"       => DType.Bool
        case "complex64"  => DType.Complex64
        case "complex128" => DType.Complex128
        case _            =>
          throw new IllegalArgumentException(s"Unsupported JAX dtype: $dtypeStr")
    catch
      case e: IllegalArgumentException => throw e
      case e: Exception                =>
        throw new RuntimeException(s"Failed to extract dtype from JAX object: ${e.getMessage}", e)

  // Convert DType to JAX dtype
  def jaxDtype(dtype: DType): py.Dynamic =
    try
      val jnp = Jax.jnp
      dtype match
        case DType.Float32    => jnp.float32
        case DType.Float64    => jnp.float64
        case DType.Int32      => jnp.int32
        case DType.Int64      => jnp.int64
        case DType.Int16      => jnp.int16
        case DType.Int8       => jnp.int8
        case DType.UInt32     => jnp.uint32
        case DType.UInt16     => jnp.uint16
        case DType.UInt8      => jnp.uint8
        case DType.Bool       => jnp.bool_
        case DType.Complex64  => jnp.complex64
        case DType.Complex128 => jnp.complex128
    catch
      case e: Exception =>
        throw new RuntimeException(s"Failed to get JAX dtype for ${dtype.name}: ${e.getMessage}", e)
