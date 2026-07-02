package dimwit.tensor.tensorops

import dimwit.linalg.LinearAlgebra
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.Tensor2
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber

object LinearAlgebraOps:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** Extracts the diagonal along the given two axes (with optional offset),
      * replacing them by a new 1D axis labeled L1.
      *
      * @see [[LinearAlgebra.diagonal]] for details
      */
    def diagonal[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes *: L1 *: EmptyTuple, V] = LinearAlgebra.diagonal(t, axis1, axis2, offset)

  extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

    /** return the diagonal of the tensor `t` along the specified axes.
      * @see [[LinearAlgebra.diagonal]] for details
      */
    def diagonal: Tensor1[L1, V] = LinearAlgebra.diagonal(t, Axis[L1], Axis[L2]).asInstanceOf[Tensor[Tuple1[L1], V]]

    /** return the diagonal of the tensor `t` along the specified axes.
      * @see [[LinearAlgebra.diagonal]] for details
      */
    def diagonal(offset: Int): Tensor1[L1, V] =
      LinearAlgebra.diagonal(t, Axis[L1], Axis[L2], offset).asInstanceOf[Tensor[Tuple1[L1], V]]

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** Computes the trace of the tensor.
      *
      * @see [[LinearAlgebra.trace]] for details
      */
    def trace[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = LinearAlgebra.trace(t, axis1, axis2, offset)

  extension [L1: Label, L2: Label, V: IsNumber](t: Tensor2[L1, L2, V])

    /** Computes the trace of the tensor
      *
      * @see [[LinearAlgebra.trace]] for details
      */
    def trace: Tensor0[V] = t.trace(0)

    /** Computes the trace. @see [[LinearAlgebra.trace]] for details */
    def trace(offset: Int): Tensor0[V] = LinearAlgebra.trace(t, Axis[L1], Axis[L2], offset)

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** Computes the element wise L2 norm of the tensor t.
      *
      * @see [[LinearAlgebra.norm]] for details
      */
    def norm: Tensor0[V] = LinearAlgebra.norm(t)

    /** @see [[LinearAlgebra.inv]] for details
      */
    def inv: Tensor[T, V] = LinearAlgebra.inv(t)

    /** Computes the determinant of the tensor `t` along the specified axes (L1, L2)
      * @see [[LinearAlgebra.det]] for details
      */
    def det[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2])(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = LinearAlgebra.det(t, axis1, axis2)

  extension [L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V])
    /** computes the determinant of the 2-D tensor t
      * @see [[LinearAlgebra.det]] for details
      */
    def det: Tensor0[V] = LinearAlgebra.det(t, Axis[L1], Axis[L2])
