package dimwit.tensor.tensorops

import dimwit.jax.Jax
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
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

object LinearAlgebraOps:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** Extracts the diagonal along the given two axes (with optional offset),
      * replacing them by a new 1D axis labeled L1.
      *
      * @param axis1 The first axis along which to extract the diagonal.
      * @param axis2 The second axis along which to extract the diagonal.
      * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
      * @return A new tensor with the diagonal extracted, where the two specified axes are replaced by a new 1D axis labeled L1.
      */
    def diagonal[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes *: L1 *: EmptyTuple, V] =
      Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

    /** return the diagonal of the tensor `t` along the specified axes.
      * The resulting 1D tensor has a single axis labeled L1, representing the diagonal index over the original (L1, L2) axes.
      *
      * @return A new tensor1 with representing the diagonal. It uses the Label of the first axis (L1) as the label for the resulting 1D tensor.
      */
    def diagonal: Tensor1[L1, V] = t.diagonal(0)

    /** return the diagonal of the tensor `t` along the specified axes.
      * The resulting 1D tensor has a single axis labeled L1, representing the diagonal index over the original (L1, L2) axes.
      *
      * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
      * @return A new tensor1 with representing the diagonal. It uses the Label of the first axis (L1) as the label for the resulting 1D tensor.
      */
    def diagonal(offset: Int): Tensor1[L1, V] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** Computes the trace of the tensor `t` along the specified axes (L1, L2) with an optional offset.
      * The resulting tensor has the two specified axes removed, and the remaining axes are preserved.
      *
      * @param axis1 The first axis along which to compute the trace.
      * @param axis2 The second axis along which to compute the trace.
      * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal, while negative values indicate diagonals below it.
      *
      * @return A new tensor with the trace computed, where the two specified axes are removed, and the remaining axes are preserved.
      */
    def trace[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  extension [L1: Label, L2: Label, V: IsNumber](t: Tensor2[L1, L2, V])

    /** Computes the trace of the tensor
      */
    def trace: Tensor0[V] = t.trace(0)

    /** Computes the trace of the tensor with an optional offset.
      *
      * @param offset The offset of the diagonal from the main diagonal. Positive values indicate diagonals above the main diagonal,
      * while negative values indicate diagonals below it.
      */
    def trace(offset: Int): Tensor0[V] = Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** Computes the L2 norm of the tensor t.
      */
    def norm: Tensor0[V] = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))

    /** Computes the inverse of the tensor t along the last two axes.
      * The first axes of the tensors are preserved, while the last
      * two axes are replaced by their inverses.
      * The tensor must be square along the last two axes.
      *
      * @return a new tensor with the same shape as t, but with the last two axes replaced by their inverses.
      */
    def inv: Tensor[T, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))

    /** Computes the determinant of the tensor `t` along the specified axes (L1, L2)
      *
      * @param axis1 The first axis along which to compute the determinant.
      * @param axis2 The second axis along which to compute the determinant.
      * @return A new tensor with the determinant computed, where the two specified axes are removed
      */
    def det[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2])(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      // JAX det only works on the last two axes (-2, -1). We must move the user's selected axes to the end.
      val moved = Jax.jnp.moveaxis(
        t.jaxValue,
        source = ev.indices.toPythonProxy,
        destination = Seq(-2, -1).toPythonProxy
      )
      Tensor(Jax.jnp.linalg.det(moved))

  extension [L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V])
    /** computes the determinant of the 2-D tensor t */
    def det: Tensor0[V] = Tensor0(Jax.jnp.linalg.det(t.jaxValue))
