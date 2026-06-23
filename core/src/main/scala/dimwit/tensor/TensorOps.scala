package dimwit.tensor

import dimwit.DType.*
import dimwit.DType.given
import dimwit.OnError
import dimwit.jax.Jax
import dimwit.tensor.HasScalar
import dimwit.tensor.{Label, Labels}
import dimwit.tensor.ShapeTypeHelpers.*
import dimwit.tensor.TensorOps.ZipVmap.{ShapesOf, TensorsOf}
import dimwit.tensor.TupleHelpers.*
import dimwit.{`|*|`, `|+|`}

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}

import scala.annotation.implicitNotFound
import scala.annotation.targetName
import scala.compiletime.ops.int.<=
import scala.util.NotGiven

import Tuple.:*
import Tuple.++
import dimwit.tensor.tensorops.StructuralOps

object TensorOps:

  import dimwit.tensor.tensorops.TensorOpsUtil.*

  /** Typeclass to map a type V to its corresponding DType.
    */
  sealed trait HasDType[V]:
    def dtype: DType

  /** Typeclass to indicate that a type V is a numeric type
    */
  @implicitNotFound("Operation only valid for Numeric (Int or Float) tensors.")
  sealed trait IsNumber[V]

  @implicitNotFound("Operation only valid for Int or Float tensors.")
  object IsNumber:
    given [V](using ev1: IsFloating[V]): IsNumber[V] = ev1
    given [V](using ev2: IsInteger[V]): IsNumber[V] = ev2

  @implicitNotFound("Operation only valid for Floating tensors.")

  /** Type class marker for floating point types (Float32, Float64, etc.). */
  trait IsFloating[V] extends IsNumber[V], HasDType[V]:
    def dtype: DType

  object IsFloating:
    def apply[V](using ev: IsFloating[V]): IsFloating[V] = ev

  /** Type class marker for integer types */
  @implicitNotFound("Operation only valid for Integer tensors.")
  trait IsInteger[V] extends IsNumber[V], HasDType[V]:
    def dtype: DType

  object IsInteger:
    def apply[V](using ev: IsInteger[V]): IsInteger[V] = ev

  /** Type class marker for Boolean types */
  @implicitNotFound("Operation only valid for Boolean tensors.")
  trait IsBoolean[V] extends HasDType[V]:
    def dtype: DType

  object IsBoolean:
    def apply[V](using ev: IsBoolean[V]): IsBoolean[V] = ev

  export tensorops.ElementWiseOps.*
  export tensorops.ReductionOps.*
  export tensorops.ContractionOps.*
  export tensorops.ConvolutionOps.*
  export tensorops.LinearAlgebraOps.*
  export tensorops.StructuralOps.*
  export tensorops.FunctionalOps.*

  export tensorops.Tensor0Ops.*
  export ValueOps.*
  export tensorops.Tensor1Ops.*
  export tensorops.Tensor2Ops.*
  export tensorops.Tensor3Ops.*

end TensorOps
