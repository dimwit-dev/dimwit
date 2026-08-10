package dimwit.tensor

import dimwit.tensor.HasScalar
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.*
import dimwit.tensor.TupleHelpers.*

import scala.annotation.implicitNotFound
import scala.annotation.targetName

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

  /** Type class marker for floating point types (Float32, Float64, etc.). */
  @implicitNotFound("Operation only valid for Floating tensors.")
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
  export tensorops.Tensor1Ops.*
  export tensorops.Tensor2Ops.*
  export tensorops.Tensor3Ops.*
