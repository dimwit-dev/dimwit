package dimwit.tensor

import scala.annotation.implicitNotFound
import me.shadaj.scalapy.py

@implicitNotFound(
  "No Scala type mapping for DType ${V}. Supported: Bool, Int8, Int16, Int32, Int64, Float32, Float64."
)
trait HasScalar[V, X]:
  def readFlat(jaxValue: py.Dynamic): Array[X]
  def classTag: scala.reflect.ClassTag[X]

object HasScalar:
  def apply[V, X](using ev: HasScalar[V, X]): HasScalar[V, X] = ev
