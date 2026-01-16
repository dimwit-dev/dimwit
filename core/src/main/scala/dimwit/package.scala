import scala.annotation.targetName

import dimwit.jax.Jax

package object dimwit:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  trait Prime[T]
  object Prime:
    given [L](using label: Label[L]): Label[Prime[L]] with
      val name: String = s"${label.name}'"

    type RemovePrimes[T <: Tuple] <: Tuple = T match
      case EmptyTuple       => EmptyTuple
      case Prime[l] *: tail => l *: RemovePrimes[tail]
      case h *: tail        => h *: RemovePrimes[tail]

    extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
      def dropPrimes: Tensor[RemovePrimes[T], V] =
        given newLabels: Labels[RemovePrimes[T]] with
          val names: List[String] =
            val oldLabels = summon[Labels[T]]
            oldLabels.names.toList.map(_.replace("'", ""))
        Tensor[RemovePrimes[T], V](tensor.jaxValue)

  def gc(): Unit =
    System.gc()
    Jax.gc()

  @targetName("On")
  infix trait ~[A, B]
  object `~`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A ~ B] with
      val name: String = s"${labelA.name}_on_${labelB.name}"

  /** Combination of dimensions / labels
    *
    * Mentally think of this as the "product" of two dimensions.
    */
  @targetName("Combined")
  infix trait |*|[A, B]
  object `|*|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |*| B] with
      val name: String = s"${labelA.name}*${labelB.name}"

  /** Concatenation of dimensions / labels
    *
    * Mentally think of this as the "sum" of two dimensions.
    */
  @targetName("Concatenated")
  infix trait |+|[A, B]
  object `|+|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |+| B] with
      val name: String = s"${labelA.name}+${labelB.name}"

  // Export tensor and related types
  export dimwit.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3}
  export dimwit.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export dimwit.tensor.{DType, Device}
  export dimwit.tensor.{VType, ExecutionType, Label, Labels, Axis, AxisIndex, AxisIndices, Dim}

  // Export operations
  export dimwit.tensor.TensorOps.*

  // Export automatic differentiation
  export dimwit.autodiff.{Autodiff, TensorTree, FloatTensorTree, ToPyTree, Grad}

  // Export Just-in-Time compilation
  export dimwit.jax.Jit.{jit, jitDonating, jitDonatingUnsafe}

  object Conversions:
    export dimwit.tensor.Tensor0.{float2FloatTensor, int2IntTensor, int2FloatTensor, boolean2BooleanTensor}
