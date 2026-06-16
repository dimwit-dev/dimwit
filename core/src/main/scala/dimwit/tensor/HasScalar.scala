package dimwit.tensor

import scala.annotation.implicitNotFound
import me.shadaj.scalapy.py

@implicitNotFound(
  "No Scala type mapping for DType ${V}. Supported: Bool, Int8, Int16, Int32, Int64, Float32, Float64."
)

/** Type class to map a DType to a Scala type for
  *   reading scalar values from JAX tensors.
  *
  *   @tparam V DType (e.g., Float32, Int32)
  *   @tparam X Corresponding Scala type (e.g., Float, Int)
  */
trait HasScalar[V, X]:

  /** read a flat array of scalar values from a JAX tensor value.
    */
  def readFlat(jaxValue: py.Dynamic): Array[X]

  /** ClassTag for the Scala type, used for array creation
    *  and pattern matching.
    */
  def classTag: scala.reflect.ClassTag[X]

object HasScalar:
  def apply[V, X](using ev: HasScalar[V, X]): HasScalar[V, X] = ev
