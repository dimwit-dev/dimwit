package dimwit.tensor.tensorops

import dimwit.tensor.Tensor
import dimwit.tensor.Labels
import dimwit.jax.Jax
import dimwit.tensor.DType.Bool
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsBoolean
import dimwit.tensor.VType
import dimwit.tensor.DType.Int32
import dimwit.tensor.DType.Float32
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast
import dimwit.tensor.Label
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.Axis
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}
import dimwit.tensor.Tensor2
import dimwit.tensor.Tensor1

object LinearAlgebraOps:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    def diagonal[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes *: L1 *: EmptyTuple, V] =
      Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

    def diagonal: Tensor1[L1, V] = t.diagonal(0)
    def diagonal(offset: Int): Tensor1[L1, V] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def trace[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
        ev: AxesRemover[T, (L1, L2)],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

  extension [L1: Label, L2: Label, V: IsNumber](t: Tensor2[L1, L2, V])

    def trace: Tensor0[V] = t.trace(0)
    def trace(offset: Int): Tensor0[V] = Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))

  // ---------------------------------------------------------
  // IsFloat operations
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    def norm: Tensor0[V] = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))
    def inv: Tensor[T, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))
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

    def det: Tensor0[V] = Tensor0(Jax.jnp.linalg.det(t.jaxValue))
